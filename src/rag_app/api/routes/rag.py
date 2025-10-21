from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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
    return RagService(
        embed_model=request.app.state.embed_model,
        vector_client=request.app.state.vector_client,
        llm_model=request.app.state.llm_model,
        encoder_model=request.app.state.reranker
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
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info("Processing query", extra={"query": request.query[:100], "request_id": request_id})
        
        result = await rag_service.answer_query(request.query.strip())
        
        # Validate response structure
        if not isinstance(result, dict) or "answer" not in result or "sources" not in result:
            logger.error(f"Invalid response from RagService: {result}")
            raise HTTPException(status_code=500, detail="Invalid service response")
        
        return AnswerResponse(
            answer=result["answer"],
            sources=[Source(**source) for source in result["sources"]]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.error(
            "Error processing query",
            extra={"error": str(e), "request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error")