from pydantic import BaseModel
from typing import Dict, Any, Optional
from enum import Enum


class IngestionStatus(str, Enum):
    PENDING = "pending"
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    EXTRACTING_METADATA = "extracting_metadata"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestRequest(BaseModel):
    source_name: str
    source_uri: str
    source_type: str
    source_config: Dict[str, Any] = {}


class IngestResponse(BaseModel):
    message: str
    task_id: str


class IngestionStatusResponse(BaseModel):
    task_id: str
    status: IngestionStatus
    message: str
    progress: Optional[int] = None  # 0-100
    error: Optional[str] = None
    total_nodes: Optional[int] = None
    processed_nodes: Optional[int] = None