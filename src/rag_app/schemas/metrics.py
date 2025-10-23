from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum


class CacheStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"


class ComponentMetrics(BaseModel):
    """Metrics for individual components in the RAG pipeline."""
    component_name: str
    duration_ms: float
    started_at: datetime
    completed_at: datetime
    metadata: Optional[Dict] = Field(default_factory=dict)


class QueryMetrics(BaseModel):
    """Comprehensive metrics for a single query."""
    query_id: str
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Timing metrics
    total_duration_ms: float
    embedding_duration_ms: float
    retrieval_duration_ms: float
    reranking_duration_ms: float
    llm_duration_ms: float
    
    # Cache metrics
    cache_status: CacheStatus
    embedding_cache_hit: bool
    semantic_cache_hit: bool
    
    # Token metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Cost metrics
    input_cost: float
    output_cost: float
    total_cost: float
    
    # Model info
    model_name: str
    provider: str
    routed: bool = False
    
    # Results
    num_documents_retrieved: int
    num_documents_reranked: int
    num_documents_final: int
    
    # Component breakdown
    components: List[ComponentMetrics] = Field(default_factory=list)


class AggregatedMetrics(BaseModel):
    """Aggregated metrics over multiple queries."""
    total_queries: int
    time_period_start: datetime
    time_period_end: datetime
    
    # Average timings
    avg_total_duration_ms: float
    avg_embedding_duration_ms: float
    avg_retrieval_duration_ms: float
    avg_reranking_duration_ms: float
    avg_llm_duration_ms: float
    
    # Cache effectiveness
    cache_hit_rate: float  # Percentage
    embedding_cache_hit_rate: float
    semantic_cache_hit_rate: float
    
    # Cost metrics
    total_cost: float
    avg_cost_per_query: float
    cost_by_provider: Dict[str, float]
    
    # Provider usage
    provider_distribution: Dict[str, int]
    
    # Performance
    queries_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class MetricsSnapshot(BaseModel):
    """A snapshot of recent metrics for the dashboard."""
    last_100_queries: List[QueryMetrics]
    aggregated: AggregatedMetrics
    generated_at: datetime = Field(default_factory=datetime.utcnow)
