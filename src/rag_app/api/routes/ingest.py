from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, BackgroundTasks
import tempfile
import os
import logging
from typing import Annotated, Dict, Any
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from rag_app.services.ingest import IngestionService
from rag_app.config.settings import settings
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

class IngestRequest(BaseModel):
    source_name: str
    source_uri: str
    source_type: str
    source_config: Dict[str, Any] = {}


class IngestResponse(BaseModel):
    message: str
    task_id: str


# Dependency for IngestionService using app state singletons
async def get_ingestion_service(request: Request) -> IngestionService:
    """Get IngestionService using app state singletons."""
    reader = PyMuPDFReader()
    node_parser = SemanticSplitterNodeParser(
        embed_model=request.app.state.chunk_embed_model,
    )
    
    return IngestionService(
        reader=reader,
        node_parser=node_parser,
        llm_model=request.app.state.llm_model,
        embedding_client=request.app.state.embed_model,
        vector_client=request.app.state.vector_client
    )

@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),    
    ingestion_service: IngestionService = Depends(get_ingestion_service)
):
    """Ingest a PDF document into the vector database."""
    
    # Enhanced validation
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Size check before reading entire file
    if hasattr(file, 'size') and file.size and file.size > settings.max_file_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size} bytes")
    
    # Read file content
    content = await file.read()
    
    # Fallback size check
    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size} bytes")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        request_id = getattr(request.state, 'request_id', 'unknown')
        logger.info(
            "Starting ingestion",
            extra={"filename": file.filename, "size": len(content), "request_id": request_id}
        )
        background_tasks.add_task(
            ingestion_service.ingest,
            source_name=request.source_name,
            source_uri=request.source_uri,
            source_type=request.source_type,
            source_config=request.source_config,
            request_id=request_id,
        )

        return IngestResponse(message="Ingestion started", task_id=request_id)
    except Exception as e:
        request_id = getattr(request.state, 'request_id', 'unknown')
        logger.error(
            "Ingestion failed",
            extra={"filename": file.filename, "error": str(e), "request_id": request_id},
            exc_info=True
        )
        raise HTTPException(status_code=422, detail=f"Ingestion error: {type(e).__name__}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)