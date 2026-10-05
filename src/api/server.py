import logging
import os
import sys
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Add the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agent.graph_agent import create_graph_agent, with_viewer_context
from src.agent.tiles import is_valid_tile_id

# uvicorn configures only its own loggers; without this the lines below are dropped
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Geospatial QA Agent API",
    description="REST API for the LangGraph ReAct agent.",
    version="1.0.0"
)

# Initialize the LangGraph Agent once when the server starts.
# MemorySaver keeps each conversation's messages in this process, keyed by thread_id,
# so history is lost on restart and is not shared between uvicorn workers.
graph_agent = create_graph_agent(checkpointer=MemorySaver())

# Define the data structure we expect from the frontend
class ChatRequest(BaseModel):
    message: str
    # Identifies the conversation. Omit it for a one-off question with no memory.
    thread_id: Optional[str] = None
    # The tile open in the viewer, so "this image" can be resolved without an ID.
    selected_tile: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    thread_id: str

@app.get("/")
def health_check():
    return {"status": "API is running", "agent": "LangGraph ReAct agent active"}

@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    """
    Receives a message from the frontend, passes it to the LangGraph ReAct agent,
    and returns the synthesized response.
    """
    # selected_tile is put into the prompt, so it must look like a tile ID and nothing else.
    # Checked outside the try block: the handler below would turn this into a 500.
    if request.selected_tile is not None and not is_valid_tile_id(request.selected_tile):
        raise HTTPException(status_code=422, detail="selected_tile is not a valid tile ID.")

    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        logger.info("thread %s, tile %s: %s", thread_id, request.selected_tile, request.message)
        content = with_viewer_context(request.message, request.selected_tile)

        # Invoke the LangGraph agent. Only the new message is sent: the checkpointer
        # loads the earlier turns for this thread_id and appends to them.
        response = graph_agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract the final AI message from the graph state
        answer = response["messages"][-1].content
        return ChatResponse(reply=answer, thread_id=thread_id)
        
    except Exception as e:
        # The full error goes to the log, not to the person asking: it can hold file paths
        # and the text of an upstream response.
        logger.exception("thread %s: the agent could not answer", thread_id)
        if getattr(e, "status_code", None) == 429 or type(e).__name__ == "RateLimitError":
            # Several people share one OpenAI quota, and questions about a cell send images
            raise HTTPException(status_code=429, detail=(
                "The model is handling too many questions right now. "
                "Wait a minute and ask again."))
        raise HTTPException(status_code=500, detail=(
            "The agent could not answer that. The error is in the API log."))

# To run this locally: uvicorn src.api.server:app --reload