import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Assuming chroma_multistep_workflow.py is in the same directory and its classes are importable
try:
    from chroma_multistep_workflow import MultiStepToolWorkflow
except ImportError:
    # Fallback/Error handling if the import path is complex
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from chroma_multistep_workflow import MultiStepToolWorkflow


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the workflow manager globally
# This ensures LLM models/embeddings are loaded only once
workflow = MultiStepToolWorkflow()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    raw_result: Dict[str, Any]

app = FastAPI(
    title="Multi-Step Tool Workflow Chat API",
    description="API to interact with the MultiStepToolWorkflow agent.",
    version="1.0.0",
)

# Enable CORS for the simple chat client HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity in demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Multi-Step Tool Workflow API is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles a chat query, optionally continuing a session.
    """
    try:
        # Run the workflow with the provided query and session ID
        result = await workflow.run_workflow(
            query=request.query,
            session_id=request.session_id
        )

        session_id = result.get("session_id")
        final_answer = result.get("final_answer")

        if not session_id:
            logger.error("Workflow failed to return a session_id.")
            raise HTTPException(status_code=500, detail="Internal server error: Missing session ID in response.")

        if not final_answer:
            error_message = result.get("error", "No response received from agent.")
            return ChatResponse(
                session_id=session_id,
                response=f"Error: {error_message}",
                raw_result=result
            )

        return ChatResponse(
            session_id=session_id,
            response=final_answer,
            raw_result=result
        )

    except Exception as e:
        logger.error(f"An error occurred during chat endpoint processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
