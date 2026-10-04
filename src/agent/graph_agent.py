import os
from typing import Optional
import duckdb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from src.agent.history import recent_turns

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
    Queries the DuckDB database to get the IoU score, Brightness, Contrast, 
    and SHAP values for a specific tile. Use this to get mathematical metrics.
    """
    if not os.path.exists(DB_PATH):
        return "Database not found."
        
    with duckdb.connect(DB_PATH) as conn:
        # tile_id comes from the LLM, so it is bound as a parameter, never formatted in
        query = "SELECT * FROM tile_metrics WHERE tile_id = ?"
        tile_data = conn.execute(query, [tile_id]).df()
    
    if tile_data.empty:
        return f"Tile {tile_id} not found in the database."
    
    row = tile_data.iloc[0]
    return f"Tile {tile_id} Metrics - IoU: {row['iou']:.4f}, Brightness SHAP impact: {row['shap_brightness']:.4f}, Contrast SHAP impact: {row['shap_contrast']:.4f}."

# --- Agent Tool 2: The Vision Annotator (GPT-4o Multimodal) ---
@tool
def run_vision_analysis(tile_id: str, specific_question: str) -> str:
    """
    Physically looks at the drone TIFF image to answer visual questions.
    Use this when you need to confirm if there are shadows, dense vegetation, or visual anomalies.
    """
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

# --- Build the LangGraph Multi-Agent Router ---
def create_graph_agent(checkpointer=None):
    """
    Pass a checkpointer to give the agent conversation memory: each invoke must then carry
    a thread_id in its config, and earlier turns on that thread are replayed to the LLM.
    Without one, every invoke is a fresh single-turn conversation (what the evals want).
    """
    # The Supervisor LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # The tools available to the Supervisor
    tools = [get_duckdb_metrics, run_vision_analysis]
    
    # System prompt dictating the routing logic
    system_prompt = """You are an expert Multi-Agent Geospatial QA Supervisor. 
    You manage two sub-agents:
    1. A Data Agent (get_duckdb_metrics) that provides mathematical IoU and SHAP values.
    2. A Vision Agent (run_vision_analysis) that can physically look at the drone imagery.
    
    When a user asks why a tile failed:
    First, use the Data Agent to get the SHAP metrics.
    Second, use the Vision Agent to look at the image and visually confirm the mathematical findings (e.g., if SHAP says brightness is an issue, ask the Vision Agent if it sees shadows).
    Finally, combine both into a comprehensive answer.

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
    again, but call the Vision Agent again when a follow-up asks about something visual it
    has not yet checked.

    The metrics decide, not the question. A user may assert that a tile failed when it did
    not. Check the IoU before accepting that framing: an IoU near 1.0 means the prediction
    matched the ground truth closely, and that tile did not fail. When the data contradicts
    the user, say so plainly in your first sentence and give the IoU, then explain what the
    numbers actually show. Never describe causes of a failure the metrics do not support,
    and never omit an IoU because it is inconvenient to the question you were asked.
    This holds after a Vision Agent call too: the answer still opens with whether the tile
    failed and its IoU, and only then reports what the Vision Agent saw. What it sees in a
    tile that scored well are conditions the model coped with, not causes of a failure."""

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
    print("🚀 Initializing LangGraph Multi-Agent System...")
    app = create_graph_agent()
    
    test_question = "Why did tile ALL-2-81-13-W6M fail? Check the data and then look at the image to confirm."
    
    print(f"\n🗣️ User: {test_question}\n")
    
    # Stream the thought process of the agents
    for chunk in app.stream({"messages": [HumanMessage(content=test_question)]}):
        if "agent" in chunk:
            print("🧠 Supervisor is thinking/routing...")
        elif "tools" in chunk:
            print("🛠️ Sub-Agent is executing a tool...")
            
    final_response = chunk["agent"]["messages"][-1].content
    print("\n✅ FINAL SYNTHESIZED ANSWER:")
    print(final_response)