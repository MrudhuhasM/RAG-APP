from contextlib import asynccontextmanager
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import CrossEncoder

from rag_app.api.routes.health import router as health
from rag_app.api.routes.root import router as root
from rag_app.api.routes.ingest import router as ingest
from rag_app.api.routes.rag import router as rag
from rag_app.config.logging import logger
from rag_app.config.settings import settings
from rag_app.embeddings import get_embed_model, get_chunk_embeddings
from rag_app.llm import get_llm_model
from rag_app.core.vector_client import VectorClient
from rag_app.core.cache_client import CacheClient
from rag_app.services.router import QueryRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - create singletons on startup."""
    logger.info("Starting RAG App - Initializing resources")
    
    # Initialize singletons
    try:
        app.state.vector_client = await VectorClient.create(
            api_key=settings.pinecone.api_key,
            environment=settings.pinecone.environment,
            index_name=settings.pinecone.index_name,
            dimension=settings.pinecone.dimension,
            metric=settings.pinecone.metric,
            cloud=settings.pinecone.cloud,
            region=settings.pinecone.region
        )

        app.state.cache_client = await CacheClient.create(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            ttl_seconds=settings.redis.ttl_seconds,
            embedding_ttl_seconds=settings.redis.embedding_ttl_seconds
        )
        
        logger.info("Loading reranker model...")
        app.state.reranker = CrossEncoder(settings.reranker.model)
        
        app.state.embed_model = get_embed_model()
        app.state.chunk_embed_model = get_chunk_embeddings()
        app.state.llm_model = get_llm_model()
        
        # Initialize query router with a classification LLM (use default LLM for classification)
        logger.info("Initializing Query Router...")
        app.state.router = QueryRouter(llm_model=app.state.llm_model)
        
        # Initialize ingestion status tracker (in-memory dict)
        app.state.ingestion_status = {}
        
        logger.info("Resources initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}", exc_info=True)
        raise
    
    yield

    if hasattr(app.state, "vector_client"):
        pass
    if hasattr(app.state, "cache_client"):
        await app.state.cache_client.close()
    
    # Cleanup on shutdown
    logger.info("Shutting down RAG App")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Add request ID middleware for observability
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except HTTPException:
        # Let HTTPExceptions propagate to return proper status codes
        raise
    except Exception as e:
        logger.error(f"Request failed", extra={"request_id": request_id, "error": str(e)}, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id}
        )


# Configure CORS based on environment
allowed_origins = (
    settings.cors_allowed_origins.split(",") 
    if settings.env == "production" 
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

app.include_router(root, prefix=API_PREFIX)
app.include_router(health, prefix=API_PREFIX)
app.include_router(ingest, prefix=API_PREFIX)
app.include_router(rag, prefix=API_PREFIX)

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML page."""
        return FileResponse(str(static_dir / "index.html"))
    
    logger.info(f"Static files mounted from: {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")