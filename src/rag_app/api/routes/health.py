from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from rag_app.config.logging import logger

router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check(request: Request):
    """
    Health check endpoint to verify that the API is running and resources are initialized.
    """
    checks = {}
    all_healthy = True
    
    # Check if vector client is initialized
    try:
        if hasattr(request.app.state, 'vector_client') and request.app.state.vector_client:
            checks["vector_db"] = "healthy"
        else:
            checks["vector_db"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        checks["vector_db"] = "unhealthy"
        all_healthy = False
    
    # Check if LLM model is initialized
    try:
        if hasattr(request.app.state, 'llm_model') and request.app.state.llm_model:
            checks["llm"] = "healthy"
        else:
            checks["llm"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        checks["llm"] = "unhealthy"
        all_healthy = False
    
    # Check if embedding model is initialized
    try:
        if hasattr(request.app.state, 'embed_model') and request.app.state.embed_model:
            checks["embeddings"] = "healthy"
        else:
            checks["embeddings"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        logger.error(f"Embeddings health check failed: {e}")
        checks["embeddings"] = "unhealthy"
        all_healthy = False
    
    # Check if reranker is initialized
    try:
        if hasattr(request.app.state, 'reranker') and request.app.state.reranker:
            checks["reranker"] = "healthy"
        else:
            checks["reranker"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        logger.error(f"Reranker health check failed: {e}")
        checks["reranker"] = "unhealthy"
        all_healthy = False
    
    status_code = 200 if all_healthy else 503
    response_data = {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks
    }
    
    return JSONResponse(status_code=status_code, content=response_data)