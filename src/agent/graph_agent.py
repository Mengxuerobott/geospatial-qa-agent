import os
import duckdb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

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
        query = f"SELECT * FROM tile_metrics WHERE tile_id = '{tile_id}'"
        tile_data = conn.execute(query).df()
    
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

# --- Build the LangGraph Multi-Agent Router ---
def create_graph_agent():
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
    Finally, combine both into a comprehensive answer."""

    # LangGraph's prebuilt ReAct agent handles the complex routing/state automatically
    graph_app = create_react_agent(llm, tools, state_modifier=system_prompt)
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