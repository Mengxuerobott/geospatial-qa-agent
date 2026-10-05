import os
from typing import Optional
import duckdb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from src.agent.cells import describe_weak_cells
from src.agent.history import recent_turns
from src.agent.tiles import is_valid_tile_id
from src.agent.verdict import FAIL_IOU_THRESHOLD, verdict

# Add vision tool import
from src.agent.vision_tool import analyze_image_visually

# --- Robust Dotenv Loading ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
env_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path=env_path)

DB_PATH = os.path.join(root_dir, "data", "metrics.duckdb")
TIFF_DIR = os.path.join(root_dir, "data", "tiffs")

# --- Agent Tool 1: The Data Analyst (DuckDB) ---
@tool
def get_duckdb_metrics(tile_id: str) -> str:
    """
    Queries the DuckDB database to get the IoU score, the pass/fail verdict and the
    SHAP values for a specific tile, plus the weak areas inside it: the grid cells where
    the model did badly. Use this to get mathematical metrics.
    """
    if not os.path.exists(DB_PATH):
        return "Database not found."
        
    with duckdb.connect(DB_PATH) as conn:
        # tile_id comes from the LLM, so it is bound as a parameter, never formatted in
        query = "SELECT * FROM tile_metrics WHERE tile_id = ?"
        tile_data = conn.execute(query, [tile_id]).df()
        try:
            cells = conn.execute(
                "SELECT * FROM cell_metrics WHERE tile_id = ?", [tile_id]).df()
        except duckdb.CatalogException:
            # A database built before the pipeline scored cells has no such table
            cells = None
    
    if tile_data.empty:
        return f"Tile {tile_id} not found in the database."
    
    row = tile_data.iloc[0]
    # The verdict is computed here rather than left to the LLM, which otherwise invents its
    # own cut-off and has called a 0.51 tile "a moderate match" that "did not fail".
    shap = ", ".join(
        f"{column[len('shap_'):].replace('_', ' ').capitalize()} SHAP impact: {row[column]:.4f}"
        for column in tile_data.columns if column.startswith("shap_")
    )
    metrics = (
        f"Tile {tile_id} Metrics - IoU: {row['iou']:.4f} "
        f"(QA verdict: {verdict(row['iou']).upper()}; tiles below {FAIL_IOU_THRESHOLD} fail), "
        f"{shap}."
    )
    if cells is None or cells.empty:
        return metrics
    return f"{metrics}\n{describe_weak_cells(cells)}"

# --- Agent Tool 2: The Vision Annotator (GPT-4o Multimodal) ---
@tool
def run_vision_analysis(tile_id: str, specific_question: str) -> str:
    """
    Physically looks at the drone TIFF image to answer visual questions.
    Use this when you need to confirm if there are shadows, dense vegetation, or visual anomalies.
    """
    # tile_id comes from the LLM and becomes part of a file path
    if not is_valid_tile_id(tile_id):
        return "That is not a valid tile ID."

    tiff_path = os.path.join(TIFF_DIR, f"{tile_id}.tif")
    if not os.path.exists(tiff_path):
        return f"Image file for {tile_id} not found."
    
    # Calls the resizer and vision LLM we just built!
    return analyze_image_visually(tiff_path, specific_question)

# --- Viewer context ---
def with_viewer_context(message: str, selected_tile: Optional[str]) -> str:
    """
    Prefix a user message with the tile open in the UI, in the format the system prompt
    describes. It goes in the message rather than the system prompt so the history records
    which tile was on screen at each turn.
    """
    if not selected_tile:
        return message
    return f"[Viewer: tile {selected_tile} is open]\n{message}"

# --- Build the LangGraph ReAct Agent ---
def create_graph_agent(checkpointer=None):
    """
    Pass a checkpointer to give the agent conversation memory: each invoke must then carry
    a thread_id in its config, and earlier turns on that thread are replayed to the LLM.
    Without one, every invoke is a fresh single-turn conversation (what the evals want).
    """
    # The agent's LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # The tools available to the agent
    tools = [get_duckdb_metrics, run_vision_analysis]
    
    # System prompt dictating which tool to call when
    system_prompt = """You are an expert Geospatial QA agent. 
    You have two tools:
    1. The metrics tool (get_duckdb_metrics), which provides mathematical IoU and SHAP values,
       and lists the weak areas of the tile: grid cells where the model did badly.
    2. The vision tool (run_vision_analysis), which can physically look at the drone imagery.
    
    When a user asks why a tile failed:
    First, use the metrics tool to get the SHAP metrics.
    Second, use the vision tool to look at the image and visually confirm the mathematical findings (e.g., if SHAP says brightness is an issue, ask the vision tool if it sees shadows).
    Finally, combine both into a comprehensive answer.
    A question about a tile's score or whether it passed needs the metrics tool only. Call the
    vision tool when the user asks why, asks for a diagnosis, or asks about the image.

    Working out which tile the user means, in this order:
    1. A tile ID written in their message always wins. Look that tile up straight away,
       even when the viewer note names a different tile: users often ask about a tile
       other than the one on screen. Do not ask them to confirm, and do not answer about
       the viewer tile instead.
    2. Otherwise, if the message starts with a "[Viewer: tile ... is open]" note, the tile
       it names is the one they are looking at on screen. "This tile", "this image", "this
       one" or a bare "why?" mean that tile, even if an earlier turn was about another.
    3. Otherwise, use the tile most recently discussed in the conversation.
    4. If none of these gives a tile, ask which one they mean. Never call a tool with a
       tile ID that did not come from the user, a viewer note or an earlier turn.
    The viewer note is added by the interface, not typed by the user; never mention it.

    Reuse metrics already in the conversation for the same tile instead of querying them
    again, but call the vision tool again when a follow-up asks about something visual it
    has not yet checked.

    A tile FAILED if its IoU is below """ + str(FAIL_IOU_THRESHOLD) + """ and PASSED otherwise. The metrics tool states the
    verdict next to the IoU; use that verdict and never substitute your own judgement of
    whether a score is good enough. A failed tile failed: do not soften it to "moderate" or
    "did not fail". A passed tile passed, however far it is from 1.0.

    The metrics decide, not the question. A user may assert that a tile failed when it
    passed, or that it did well when it failed. Check the verdict before accepting their
    framing. When the data contradicts the user, say so plainly in your first sentence and
    give the IoU, then explain what the numbers actually show. When the data agrees with the
    user, say that just as plainly. Never describe causes of a failure the metrics do not
    support, and never omit an IoU because it is inconvenient to the question you were asked.
    This holds after a vision tool call too: the answer still opens with whether the tile
    failed and its IoU, and only then reports what the vision tool saw. What it sees in a
    tile that scored well are conditions the model coped with, not causes of a failure.

    A tile that passed can still have weak areas, and a tile that failed has usually failed
    in some places more than others. The metrics tool lists them. Report weak areas only
    when it does, after the verdict and the IoU, and say where they are. They never change
    the verdict: a passed tile with weak areas still passed. Give a weak area's score as a
    percentage, as the tool does, and never call it an IoU; the IoU is the tile's alone.
    A question that asks only for a tile's score needs one sentence on its weak areas, not
    the list. When you call the vision tool about a tile that has weak areas, tell it where
    they are so it looks there."""

    # LangGraph's prebuilt ReAct agent handles the complex routing/state automatically
    # Only the last few turns are sent to the LLM, so a long chat does not keep growing the
    # prompt. The checkpointer still stores the full history; this bounds what is replayed.
    def build_prompt(state):
        recent = recent_turns(state["messages"])
        prompt = [SystemMessage(content=system_prompt)]
        dropped = sum(isinstance(m, HumanMessage) for m in state["messages"]) - sum(
            isinstance(m, HumanMessage) for m in recent
        )
        if dropped:
            # Otherwise the LLM takes the oldest turn it can see for the start of the chat.
            prompt.append(SystemMessage(content=(
                f"[The first {dropped} question(s) of this conversation and their answers "
                "were removed here to save space. The next message is question "
                f"{dropped + 1}, not the first. If the user asks about the removed part, "
                "say you no longer have it.]"
            )))
        return prompt + recent

    graph_app = create_react_agent(
        llm, tools, state_modifier=build_prompt, checkpointer=checkpointer
    )
    return graph_app

# --- Test the Graph ---
if __name__ == "__main__":
    print("🚀 Initializing LangGraph ReAct agent...")
    app = create_graph_agent()
    
    test_question = "Why did tile ALL-2-81-13-W6M fail? Check the data and then look at the image to confirm."
    
    print(f"\n🗣️ User: {test_question}\n")
    
    # Stream the thought process of the agent
    for chunk in app.stream({"messages": [HumanMessage(content=test_question)]}):
        if "agent" in chunk:
            print("🧠 Agent is thinking...")
        elif "tools" in chunk:
            print("🛠️ Agent is executing a tool...")
            
    final_response = chunk["agent"]["messages"][-1].content
    print("\n✅ FINAL SYNTHESIZED ANSWER:")
    print(final_response)