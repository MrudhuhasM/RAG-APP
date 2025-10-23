# Performance Metrics

**Last Updated:** 2025-10-23 17:22:14

📊 **Status:** System ready - metrics will appear after queries are processed

---

## Executive Summary

This document will be populated with real performance metrics once queries are processed through the system.

### Key Performance Indicators (Target Values)

| Metric | Target | Notes |
|--------|--------|-------|
| **Average Latency** | <5000ms | End-to-end query processing |
| **P95 Latency** | <8000ms | 95th percentile response time |
| **Cache Hit Rate** | >30% | Combined exact + semantic cache |
| **Average Cost/Query** | <$0.01 | With intelligent routing |
| **Throughput** | >0.5 QPS | Single instance capacity |

---

## Latency Breakdown

This section shows where time is spent during query processing.

### Component Timing (Expected)

| Component | Expected Time | Notes |
|-----------|---------------|-------|
| **Embedding Generation** | 200-500ms | Depends on model and caching |
| **Document Retrieval** | 50-150ms | Vector search in Pinecone |
| **Reranking** | 300-800ms | CrossEncoder on CPU |
| **LLM Generation** | 2000-4000ms | Varies by model and length |
| **Total** | 3000-6000ms | End-to-end processing |

---

## Cache Effectiveness

Multi-tier caching strategy significantly improves performance and reduces costs.

### Cache Strategy

The system implements three cache levels:

1. **Exact Match Cache (Redis)**
   - Instant retrieval for identical queries
   - Expected hit rate: 20-30%

2. **Semantic Cache (Vector-based)**
   - Matches similar queries (>85% similarity)
   - Expected hit rate: 10-15%

3. **Embedding Cache (Redis)**
   - Caches query embeddings
   - Expected hit rate: 40-60%

---

## Cost Analysis

Intelligent query routing optimizes cost while maintaining quality.

### Expected Cost Structure

| Model | Cost per 1K tokens (Input) | Cost per 1K tokens (Output) | Use Case |
|-------|---------------------------|----------------------------|----------|
| **Gemini 2.5 Flash** | $0.000075 | $0.0003 | Simple queries, high volume |
| **Gemini 2.5 Pro** | $0.00125 | $0.005 | Complex analysis |
| **GPT-4o** | $0.0025 | $0.01 | Critical, complex reasoning |

**Routing Strategy:**
- Simple queries → Gemini Flash (~60% of queries)
- Medium complexity → Gemini Pro (~30% of queries)
- Complex/critical → GPT-4o (~10% of queries)

**Expected Savings:** 60-70% vs. using GPT-4o for all queries

---

## Throughput & Scalability

### Capacity Planning

**Single Instance (Cloud Run):**
- Expected capacity: 0.5-2 QPS (depending on cache hit rate)
- Memory: ~512MB-1GB per instance
- CPU: 1-2 vCPU recommended

**Scaling Strategy:**
- Horizontal scaling via Cloud Run auto-scaling
- Cache hit rate improves with higher load
- Pinecone handles vector search scaling independently

---

## System Health Indicators

### Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| **Availability** | >99% | {'✅' if has_real_data else 'ℹ️'} Monitoring active |
| **Error Rate** | <1% | {'✅' if has_real_data else 'ℹ️'} To be measured |
| **Cache Hit Rate** | >30% | {'✅ ' + f'{metrics.cache_hit_rate:.1f}%' if has_real_data and metrics.cache_hit_rate > 30 else 'ℹ️ Pending data'} |
| **P95 Latency** | <8s | {'✅ ' + f'{metrics.p95_latency_ms/1000:.1f}s' if has_real_data and metrics.p95_latency_ms < 8000 else 'ℹ️ Pending data'} |

---

## Optimization Roadmap

### Completed ✅
- Multi-tier caching implementation
- Intelligent query routing
- Performance monitoring infrastructure
- Real-time metrics collection

### In Progress 🚧
- Baseline performance documentation
- Load testing under concurrent users
- Cost optimization analysis

### Planned 📋
- Batch embedding requests for multiple queries
- Request coalescing for similar concurrent queries  
- CUDA-accelerated reranking (Phase 2)
- Custom kernel optimizations (Phase 2)

---

## How to Reproduce These Metrics

```bash
# 1. Start the RAG application
uv run fastapi dev src/rag_app/app.py

# 2. Run benchmark queries
uv run python scripts/benchmark.py

# 3. View metrics endpoint
curl http://localhost:8000/api/v1/metrics

# 4. Check this document for updated metrics
cat PERFORMANCE.md
```

---

## Monitoring & Observability

### Available Endpoints

- **Metrics API:** `/api/v1/metrics` - Real-time performance data
- **Health Check:** `/api/v1/health` - System status
- **API Docs:** `/docs` - Interactive API documentation

### Logging

All queries are logged with:
- Request ID for tracing
- Component-level timing
- Token usage and costs
- Cache hit/miss information
- Model selection reasoning

---

**Note:** This document is automatically updated after running benchmarks. For the latest metrics, run `uv run python scripts/benchmark.py`.
