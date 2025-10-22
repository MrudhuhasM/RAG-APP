import json
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.responses import StreamingResponse
from rag_app.services.rag import RagService
from rag_app.config.settings import settings
from rag_app.config.logging import logger

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    content: str
    score: float
    source: str
    rerank_score: Optional[float] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]

# Dependency for RagService using app state singletons
async def get_rag_service(request: Request) -> RagService:
    """Get RagService using app state singletons."""
    
    if request.app.state.cache_client is None:
        logger.error("Cache client not initialized in app state")
        raise HTTPException(status_code=500, detail="Cache client not initialized")


    return RagService(
        embed_model=request.app.state.embed_model,
        vector_client=request.app.state.vector_client,
        llm_model=request.app.state.llm_model,
        encoder_model=request.app.state.reranker,
        cache_client=request.app.state.cache_client

    )

@router.post("/query", response_model=AnswerResponse, tags=["RAG"])
async def answer_query(
    request: QueryRequest,
    http_request: Request,
    rag_service: RagService = Depends(get_rag_service)
):
    """
    Answer a user query using Retrieval-Augmented Generation (RAG).
    
    - **query**: The user's question to be answered.
    
    Returns the generated answer along with relevant sources.
    """
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def stream_generator():
        try:
            async for chunk in rag_service.answer_query(request.query.strip()):
                yield f"data: {json.dumps(chunk)}\n\n"  
        except Exception as e:
            logger.error(f"Error in stream_generator: {e}", exc_info=True)
            error_event = {"type": "error", "data": "An error occurred while processing your query."}
            yield f"data: {json.dumps(error_event)}\n\n"
            
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
    