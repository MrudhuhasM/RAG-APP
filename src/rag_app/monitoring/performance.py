import time
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
import statistics
import uuid

from rag_app.schemas.metrics import (
    QueryMetrics,
    ComponentMetrics,
    AggregatedMetrics,
    MetricsSnapshot,
    CacheStatus
)
from rag_app.config.logging import logger


class PerformanceTracker:
    """Tracks performance metrics for RAG queries. Thread-safe, singleton pattern for application-wide usage."""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.query_history: deque[QueryMetrics] = deque(maxlen=max_history)
        self._current_query_context: Dict[str, Any] = {}
        
    @classmethod
    async def get_instance(cls, max_history: int = 1000):
        """Get or create the singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_history)
                    logger.info(f"Performance tracker initialized with max_history={max_history}")
        return cls._instance
    
    @asynccontextmanager
    async def track_query(self, query: str, query_id: Optional[str] = None):
        """Context manager to track a complete query execution."""
        if query_id is None:
            query_id = str(uuid.uuid4())
        
        context = QueryContext(query_id, query)
        
        try:
            yield context
        finally:
            metrics = context.finalize()
            await self.record_query(metrics)
    
    async def record_query(self, metrics: QueryMetrics):
        """Record a completed query metrics."""
        async with self._lock:
            self.query_history.append(metrics)
            logger.info(
                f"Query recorded: {metrics.query_id} | "
                f"Duration: {metrics.total_duration_ms:.2f}ms | "
                f"Model: {metrics.model_name} | "
                f"Cost: ${metrics.total_cost:.6f} | "
                f"Cache: {metrics.cache_status.value}"
            )
    
    async def get_recent_queries(self, limit: int = 100) -> List[QueryMetrics]:
        """Get the most recent query metrics."""
        async with self._lock:
            return list(self.query_history)[-limit:]
    
    async def get_aggregated_metrics(self, time_window_minutes: Optional[int] = None) -> AggregatedMetrics:
        """Calculate aggregated metrics over the specified time window."""
        async with self._lock:
            queries = list(self.query_history)
        
        if not queries:
            return self._empty_aggregated_metrics()
        
        if time_window_minutes:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            queries = [q for q in queries if q.timestamp >= cutoff_time]
        
        if not queries:
            return self._empty_aggregated_metrics()
        
        return self._calculate_aggregated_metrics(queries)
    
    def _empty_aggregated_metrics(self) -> AggregatedMetrics:
        """Return empty aggregated metrics."""
        return AggregatedMetrics(
            total_queries=0,
            time_period_start=datetime.utcnow(),
            time_period_end=datetime.utcnow(),
            avg_total_duration_ms=0.0,
            avg_embedding_duration_ms=0.0,
            avg_retrieval_duration_ms=0.0,
            avg_reranking_duration_ms=0.0,
            avg_llm_duration_ms=0.0,
            cache_hit_rate=0.0,
            embedding_cache_hit_rate=0.0,
            semantic_cache_hit_rate=0.0,
            total_cost=0.0,
            avg_cost_per_query=0.0,
            cost_by_provider={},
            provider_distribution={},
            queries_per_second=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0
        )
    
    def _calculate_aggregated_metrics(self, queries: List[QueryMetrics]) -> AggregatedMetrics:
        """Calculate aggregated metrics from a list of queries."""
        total_queries = len(queries)
        time_period_start = min(q.timestamp for q in queries)
        time_period_end = max(q.timestamp for q in queries)
        
        avg_total_duration = statistics.mean(q.total_duration_ms for q in queries)
        avg_embedding_duration = statistics.mean(q.embedding_duration_ms for q in queries)
        avg_retrieval_duration = statistics.mean(q.retrieval_duration_ms for q in queries)
        avg_reranking_duration = statistics.mean(q.reranking_duration_ms for q in queries)
        avg_llm_duration = statistics.mean(q.llm_duration_ms for q in queries)
        
        cache_hits = sum(1 for q in queries if q.cache_status == CacheStatus.HIT)
        embedding_cache_hits = sum(1 for q in queries if q.embedding_cache_hit)
        semantic_cache_hits = sum(1 for q in queries if q.semantic_cache_hit)
        
        cache_hit_rate = (cache_hits / total_queries) * 100
        embedding_cache_hit_rate = (embedding_cache_hits / total_queries) * 100
        semantic_cache_hit_rate = (semantic_cache_hits / total_queries) * 100
        
        total_cost = sum(q.total_cost for q in queries)
        avg_cost = total_cost / total_queries
        
        cost_by_provider: Dict[str, float] = {}
        provider_distribution: Dict[str, int] = {}
        for q in queries:
            cost_by_provider[q.provider] = cost_by_provider.get(q.provider, 0.0) + q.total_cost
            provider_distribution[q.provider] = provider_distribution.get(q.provider, 0) + 1
        
        time_span = (time_period_end - time_period_start).total_seconds()
        qps = total_queries / time_span if time_span > 0 else 0.0
        
        latencies = sorted(q.total_duration_ms for q in queries)
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        
        p50_latency = latencies[p50_idx] if p50_idx < len(latencies) else 0.0
        p95_latency = latencies[p95_idx] if p95_idx < len(latencies) else 0.0
        p99_latency = latencies[p99_idx] if p99_idx < len(latencies) else 0.0
        
        return AggregatedMetrics(
            total_queries=total_queries,
            time_period_start=time_period_start,
            time_period_end=time_period_end,
            avg_total_duration_ms=avg_total_duration,
            avg_embedding_duration_ms=avg_embedding_duration,
            avg_retrieval_duration_ms=avg_retrieval_duration,
            avg_reranking_duration_ms=avg_reranking_duration,
            avg_llm_duration_ms=avg_llm_duration,
            cache_hit_rate=cache_hit_rate,
            embedding_cache_hit_rate=embedding_cache_hit_rate,
            semantic_cache_hit_rate=semantic_cache_hit_rate,
            total_cost=total_cost,
            avg_cost_per_query=avg_cost,
            cost_by_provider=cost_by_provider,
            provider_distribution=provider_distribution,
            queries_per_second=qps,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency
        )
    
    async def get_snapshot(self, limit: int = 100) -> MetricsSnapshot:
        """Get a complete metrics snapshot for the dashboard."""
        recent_queries = await self.get_recent_queries(limit)
        aggregated = await self.get_aggregated_metrics()
        
        return MetricsSnapshot(
            last_100_queries=recent_queries,
            aggregated=aggregated
        )
    
    async def clear_history(self):
        """Clear all query history (use with caution)."""
        async with self._lock:
            self.query_history.clear()
            logger.warning("Performance tracker history cleared")


class QueryContext:
    """Context object for tracking a single query execution."""
    
    def __init__(self, query_id: str, query: str):
        self.query_id = query_id
        self.query = query
        self.start_time = time.time()
        
        self.embedding_duration_ms = 0.0
        self.retrieval_duration_ms = 0.0
        self.reranking_duration_ms = 0.0
        self.llm_duration_ms = 0.0
        
        self.cache_status = CacheStatus.MISS
        self.embedding_cache_hit = False
        self.semantic_cache_hit = False
        
        self.input_tokens = 0
        self.output_tokens = 0
        
        self.model_name = "unknown"
        self.provider = "unknown"
        self.routed = False
        
        self.num_documents_retrieved = 0
        self.num_documents_reranked = 0
        self.num_documents_final = 0
        
        self.components: List[ComponentMetrics] = []
        
        self.input_cost = 0.0
        self.output_cost = 0.0
        self.total_cost = 0.0
    
    @asynccontextmanager
    async def track_component(self, component_name: str, metadata: Optional[Dict] = None):
        """Track timing for a specific component."""
        started_at = datetime.utcnow()
        start_time = time.time()
        
        try:
            yield
        finally:
            completed_at = datetime.utcnow()
            duration_ms = (time.time() - start_time) * 1000
            
            component_metric = ComponentMetrics(
                component_name=component_name,
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
                metadata=metadata or {}
            )
            self.components.append(component_metric)
            
            if component_name == "embedding":
                self.embedding_duration_ms += duration_ms
            elif component_name == "retrieval":
                self.retrieval_duration_ms += duration_ms
            elif component_name == "reranking":
                self.reranking_duration_ms += duration_ms
            elif component_name == "llm":
                self.llm_duration_ms += duration_ms
    
    def set_cache_status(self, status: CacheStatus):
        self.cache_status = status
    
    def set_embedding_cache_hit(self, hit: bool):
        self.embedding_cache_hit = hit
    
    def set_semantic_cache_hit(self, hit: bool):
        self.semantic_cache_hit = hit
    
    def set_tokens(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
    
    def set_model(self, model_name: str, provider: str = "unknown", routed: bool = False):
        self.model_name = model_name
        self.provider = provider
        self.routed = routed
    
    def set_document_counts(self, retrieved: int, reranked: int, final: int):
        self.num_documents_retrieved = retrieved
        self.num_documents_reranked = reranked
        self.num_documents_final = final
    
    def calculate_cost(self, model_costs: Optional[Dict[str, Dict[str, float]]] = None):
        """Calculate cost based on token usage and model pricing."""
        if model_costs and self.model_name in model_costs:
            costs = model_costs[self.model_name]
            self.input_cost = (self.input_tokens / 1000) * costs.get('input', 0)
            self.output_cost = (self.output_tokens / 1000) * costs.get('output', 0)
            self.total_cost = self.input_cost + self.output_cost
        else:
            logger.warning(f"No cost data available for model: {self.model_name}")
    
    def finalize(self) -> QueryMetrics:
        """Finalize and return the complete metrics."""
        total_duration_ms = (time.time() - self.start_time) * 1000
        
        return QueryMetrics(
            query_id=self.query_id,
            query=self.query,
            timestamp=datetime.utcnow(),
            total_duration_ms=total_duration_ms,
            embedding_duration_ms=self.embedding_duration_ms,
            retrieval_duration_ms=self.retrieval_duration_ms,
            reranking_duration_ms=self.reranking_duration_ms,
            llm_duration_ms=self.llm_duration_ms,
            cache_status=self.cache_status,
            embedding_cache_hit=self.embedding_cache_hit,
            semantic_cache_hit=self.semantic_cache_hit,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            input_cost=self.input_cost,
            output_cost=self.output_cost,
            total_cost=self.total_cost,
            model_name=self.model_name,
            provider=self.provider,
            routed=self.routed,
            num_documents_retrieved=self.num_documents_retrieved,
            num_documents_reranked=self.num_documents_reranked,
            num_documents_final=self.num_documents_final,
            components=self.components
        )
