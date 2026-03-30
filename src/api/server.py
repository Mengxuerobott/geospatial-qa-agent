import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agent.graph_agent import create_graph_agent

# Initialize FastAPI App
app = FastAPI(
    title="Geospatial QA Agent API",
    description="REST API for the LangGraph Multi-Agent system.",
    version="1.0.0"
)

# Initialize the LangGraph Agent once when the server starts
graph_agent = create_graph_agent()

# Define the data structure we expect from the frontend
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def health_check():
    return {"status": "API is running", "agent": "LangGraph Multi-Agent Active"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    """
    Receives a message from the frontend, passes it to the LangGraph Multi-Agent,
    and returns the synthesized response.
    """
    try:
        print(f"📩 Received message: {request.message}")
        
        # Invoke the LangGraph agent
        response = graph_agent.invoke(
            {"messages": [HumanMessage(content=request.message)]}
        )
        
        # Extract the final AI message from the graph state
        answer = response["messages"][-1].content
        return ChatResponse(reply=answer)
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run this locally: uvicorn src.api.server:app --reload