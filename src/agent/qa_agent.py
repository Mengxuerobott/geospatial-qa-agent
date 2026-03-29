import os
import duckdb
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# --- Robust Dotenv Loading ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
env_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(f"CRITICAL: OPENAI_API_KEY not found. Checked path: {env_path}")

# --- Path to your new DuckDB ---
DB_PATH = os.path.join(root_dir, "data", "metrics.duckdb")

# --- 2. Define the Tools for the LLM (NOW USING DUCKDB) ---

@tool
def get_worst_tiles(limit: int = 3) -> str:
    """
    Queries the DuckDB database to find the tiles with the lowest IoU scores.
    Use this when the user asks 'Which tiles performed the worst?'
    """
    if not os.path.exists(DB_PATH):
        return "Database not found. Please run the data pipeline first."
        
    with duckdb.connect(DB_PATH) as conn:
        # Get the rows with the lowest IoU
        query = f"SELECT tile_id, iou, brightness, contrast FROM tile_metrics ORDER BY iou ASC LIMIT {limit}"
        worst_tiles = conn.execute(query).df()
        
    result = worst_tiles.to_dict(orient='records')
    return f"The worst {limit} tiles are: {result}"

@tool
def explain_tile_failure(tile_id: str) -> str:
    """
    Analyzes the SHAP values from the database to explain exactly WHY a specific tile failed.
    Use this when the user asks 'Why did tile X fail?'
    """
    if not os.path.exists(DB_PATH):
        return "Database not found."
        
    with duckdb.connect(DB_PATH) as conn:
        query = f"SELECT * FROM tile_metrics WHERE tile_id = '{tile_id}'"
        tile_data = conn.execute(query).df()
    
    if tile_data.empty:
        return f"Tile {tile_id} not found in the database."
    
    row = tile_data.iloc[0]
    iou = row['iou']
    shap_bright = row['shap_brightness']
    shap_cont = row['shap_contrast']
    
    explanation = f"Tile '{tile_id}' has an IoU of {iou:.2f}. "
    
    # Let the LLM read the SHAP logic!
    if shap_bright > 0:
        explanation += f"Brightness increased the error rate (SHAP impact: +{shap_bright:.4f}). "
    else:
        explanation += f"Brightness reduced the error rate (SHAP impact: {shap_bright:.4f}). "
        
    if shap_cont > 0:
        explanation += f"Contrast increased the error rate (SHAP impact: +{shap_cont:.4f}). "
    else:
        explanation += f"Contrast reduced the error rate (SHAP impact: {shap_cont:.4f}). "
        
    return explanation

# --- 3. Build the Agent ---
def create_qa_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_worst_tiles, explain_tile_failure]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Geospatial AI QA Assistant. 
        Your job is to help AI engineers triage and understand model failures on drone imagery.
        You have access to tools that query a DuckDB database containing IoU scores and SHAP explainability metrics.
        Always use your tools to answer data-specific questions. Translate SHAP impacts into plain English."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Test the Agent ---
if __name__ == "__main__":
    print("Initializing Agent...\n")
    qa_agent = create_qa_agent()
    
    # Test Question 1: Finding bad tiles
    print("\n--- Question 1 ---")
    response1 = qa_agent.invoke({"input": "Which 2 tiles have the worst predictions and need review?"})
    print("\nFinal Answer:\n", response1['output'])
    
    # Test Question 2: Explaining a specific failure
    print("\n--- Question 2 ---")
    response2 = qa_agent.invoke({"input": "Can you explain why tile_004 failed so badly?"})
    print("\nFinal Answer:\n", response2['output'])