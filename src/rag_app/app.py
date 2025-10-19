from rag_app.config.logging import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_app.api.routes.health import router as health
from rag_app.api.routes.root import router as root
from rag_app.config.logging import logger
from rag_app.config.settings import settings

logger.info("Starting RAG App API")

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

app.include_router(root, prefix=API_PREFIX)
app.include_router(health, prefix=API_PREFIX)