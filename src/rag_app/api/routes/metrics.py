"""
Performance metrics API endpoints.

Provides access to real-time performance metrics including:
- Query latencies and component breakdowns
- Cache hit rates
- Cost tracking per provider
- Throughput metrics
"""

from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional

from rag_app.monitoring import PerformanceTracker
from rag_app.schemas.metrics import MetricsSnapshot, AggregatedMetrics, QueryMetrics
from rag_app.config.logging import logger

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=MetricsSnapshot)
async def get_metrics_snapshot(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000, description="Number of recent queries to include"),
    time_window_minutes: Optional[int] = Query(default=None, ge=1, description="Time window for aggregated metrics (minutes)")
):
    """
    Get a comprehensive metrics snapshot.
    
    Returns:
    - Last N queries with detailed metrics
    - Aggregated statistics (averages, percentiles, cache rates, costs)
    - Provider distribution
    
    **Parameters:**
    - `limit`: Number of recent queries to include (1-1000, default: 100)
    - `time_window_minutes`: Only aggregate metrics from the last N minutes (optional)
    
    **Example Response:**
    ```json
    {
        "last_100_queries": [...],
        "aggregated": {
            "total_queries": 250,
            "avg_total_duration_ms": 4523.5,
            "cache_hit_rate": 35.2,
            "avg_cost_per_query": 0.00456,
            "provider_distribution": {
                "gemini-flash": 150,
                "gemini-pro": 75,
                "gpt-4o": 25
            },
            ...
        },
        "generated_at": "2025-10-23T16:48:00Z"
    }
    ```
    """
    try:
        # Get performance tracker from app state
        if not hasattr(request.app.state, 'performance_tracker'):
            raise HTTPException(
                status_code=503,
                detail="Performance tracking not initialized"
            )
        
        tracker: PerformanceTracker = request.app.state.performance_tracker
        
        # Get snapshot
        snapshot = await tracker.get_snapshot(limit=limit)
        
        # If time window specified, recalculate aggregated metrics
        if time_window_minutes:
            snapshot.aggregated = await tracker.get_aggregated_metrics(time_window_minutes=time_window_minutes)
        
        logger.info(f"Metrics snapshot generated: {len(snapshot.last_100_queries)} queries, window={time_window_minutes}min")
        
        return snapshot
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating metrics snapshot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate metrics snapshot")


@router.get("/metrics/aggregated", response_model=AggregatedMetrics)
async def get_aggregated_metrics(
    request: Request,
    time_window_minutes: Optional[int] = Query(default=None, ge=1, description="Time window in minutes")
):
    """
    Get aggregated performance metrics.
    
    Returns summary statistics across all queries (or within a time window):
    - Average latencies per component
    - Cache effectiveness rates
    - Cost totals and averages
    - Provider distribution
    - Percentile latencies (P50, P95, P99)
    
    **Parameters:**
    - `time_window_minutes`: Only include queries from the last N minutes (optional)
    
    **Use Cases:**
    - Dashboard displays
    - Monitoring alerts
    - Performance trend analysis
    """
    try:
        if not hasattr(request.app.state, 'performance_tracker'):
            raise HTTPException(
                status_code=503,
                detail="Performance tracking not initialized"
            )
        
        tracker: PerformanceTracker = request.app.state.performance_tracker
        metrics = await tracker.get_aggregated_metrics(time_window_minutes=time_window_minutes)
        
        logger.info(f"Aggregated metrics generated: {metrics.total_queries} queries, window={time_window_minutes}min")
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating aggregated metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate aggregated metrics")


@router.get("/metrics/recent", response_model=list[QueryMetrics])
async def get_recent_queries(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100, description="Number of recent queries")
):
    """
    Get detailed metrics for recent queries.
    
    Returns individual query metrics with full component breakdowns,
    useful for debugging and detailed analysis.
    
    **Parameters:**
    - `limit`: Number of recent queries (1-100, default: 10)
    
    **Response includes for each query:**
    - Query text and ID
    - Total duration and component breakdowns
    - Cache status (hit/miss)
    - Token usage and costs
    - Model used and routing decision
    - Document counts (retrieved/reranked/final)
    """
    try:
        if not hasattr(request.app.state, 'performance_tracker'):
            raise HTTPException(
                status_code=503,
                detail="Performance tracking not initialized"
            )
        
        tracker: PerformanceTracker = request.app.state.performance_tracker
        queries = await tracker.get_recent_queries(limit=limit)
        
        logger.info(f"Recent queries retrieved: {len(queries)} queries")
        
        return queries
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving recent queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve recent queries")


@router.delete("/metrics/clear")
async def clear_metrics_history(request: Request):
    """
    Clear all metrics history.
    
    ⚠️ **WARNING:** This action cannot be undone!
    
    Use cases:
    - Resetting metrics after configuration changes
    - Clearing test data before production launch
    - Starting fresh benchmark runs
    
    **Requires:** Admin access (implement authentication as needed)
    """
    try:
        if not hasattr(request.app.state, 'performance_tracker'):
            raise HTTPException(
                status_code=503,
                detail="Performance tracking not initialized"
            )
        
        tracker: PerformanceTracker = request.app.state.performance_tracker
        await tracker.clear_history()
        
        logger.warning("Metrics history cleared via API")
        
        return {
            "status": "success",
            "message": "All metrics history has been cleared",
            "timestamp": "2025-10-23T16:48:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear metrics")
