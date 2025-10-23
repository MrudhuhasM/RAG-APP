# RAG System with Intelligent Query Routing

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval-Augmented Generation (RAG)** system that processes technical documentation and provides accurate answers using state-of-the-art LLMs, semantic caching, and adaptive query routing. Built as a production prototype demonstrating scalable architecture patterns while maintaining simplicity.



---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Performance Metrics](#performance-metrics)
- [Performance Dashboard](#performance-dashboard)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)
- [Testing](#testing)

---

## Overview

This RAG system demonstrates production-grade engineering practices in a focused prototype. It showcases intelligent document processing, multi-tier caching strategies, and adaptive LLM selection while keeping the implementation clean and maintainable.

**Project Philosophy:**
- **Extensibility over features**: Built with abstract base classes and plugin-ready architecture
- **Simplicity over complexity**: Currently supports PDF only, but designed for easy multi-format extension
- **Production patterns**: Implements industry best practices without over-engineering

---

## Key Features

## Key Features

### Adaptive Query Routing
- LLM-powered complexity classification routes queries to optimal models
- Simple queries → lightweight local models; Complex queries → advanced cloud models
- Reduces inference costs by ~60-70% while maintaining answer quality
- Graceful fallback on routing failures

### Multi-Tier Caching Strategy
- **Semantic Cache**: Vector-based similarity matching (>85% threshold) finds cached results for semantically similar queries
- **Exact Match Cache**: Redis-backed key-value store for identical queries  
- **Embedding Cache**: Reuses embeddings across requests
- Cache hit latency: <500ms (vs 3-7s cold start)

### Intelligent Retrieval Pipeline
- **Hybrid Search**: Combines document chunks (top-10) with generated questions (top-20)
- **Cross-Encoder Reranking**: BAAI/bge-reranker-base scores and reorders candidates
- **Query Rewriting**: LLM-enhanced query expansion improves retrieval recall
- **Token-Aware Context**: Dynamically truncates context to fit LLM limits (5000 tokens)

### Document Processing
- **PDF Support**: PyMuPDF for robust parsing (extensible to DOCX, HTML, Markdown via factory pattern)
- **Intelligent Chunking**: LlamaIndex with respect for semantic boundaries
- **Multi-Provider Embeddings**: Supports OpenAI, Gemini, and local models via abstract interfaces
- **Batch Upserts**: Configurable batch sizes (default: 100) for efficient vector storage

### Real-Time Streaming
- Server-Sent Events (SSE) for token-by-token answer generation
- Progressive UI updates with status notifications
- Source attribution with relevance scores

### Production Engineering
- **Resource Management**: Singleton pattern prevents per-request initialization overhead
- **Retry Logic**: Exponential backoff (3 attempts, 4-10s delays) on transient failures
- **Observability**: Request tracing, structured logging, component health checks
- **Performance Monitoring**: Real-time metrics tracking with automated benchmarking
- **Interactive Dashboard**: Visual performance analytics at `/dashboard`
- **12-Factor Compliance**: Environment-based configuration, stateless compute

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│           (HTML/CSS/JS + SSE Streaming)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Ingestion  │  │   RAG Query  │  │  Health/Status  │   │
│  │   Service    │  │   Service    │  │    Endpoints    │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└───────────┬────────────────┬───────────────────────────────┘
            │                │
    ┌───────▼──────┐  ┌─────▼──────────────────┐
    │   Document   │  │    Query Router        │
    │  Processing  │  │  (Complexity Analysis) │
    └───────┬──────┘  └─────┬──────────────────┘
            │                │
    ┌───────▼────────────────▼───────────┐
    │      Embedding Generation          │
    │  (OpenAI / Gemini / Local)        │
    └───────┬────────────────────────────┘
            │
    ┌───────▼────────────────────────────┐
    │       Vector Database              │
    │         (Pinecone)                 │
    │  • Main Index (Documents)          │
    │  • Questions Namespace             │
    │  • Semantic Cache Namespace        │
    └────────────────────────────────────┘
    
    ┌────────────────────────────────────┐
    │      Redis Cache Layer             │
    │  • Exact Match Cache               │
    │  • Embedding Cache                 │
    │  • TTL-based Expiration            │
    └────────────────────────────────────┘
    
    ┌────────────────────────────────────┐
    │      LLM Inference                 │
    │  • Local Models (Qwen)             │
    │  • OpenAI (GPT-4/3.5)              │
    │  • Google Gemini (2.0 Flash)       │
    └────────────────────────────────────┘
```

**Data Flow:**
1. **Ingestion**: PDF → Parse → Chunk → Embed → Store in Pinecone + Cache in Redis
2. **Query**: Question → Rewrite → Embed → Multi-Tier Cache Check → Vector Search (Documents + Questions) → Dedup → Rerank → Route to LLM → Generate Answer → Cache Result

---

## Quick Start

## Quick Start

### Prerequisites
- **Python**: ≥ 3.12
- **uv**: Modern Python package manager ([installation](https://github.com/astral-sh/uv))
- **Redis**: Optional but recommended for caching
- **API Keys**: OpenAI and Pinecone

### Installation

```bash
# Clone the repository
git clone https://github.com/MrudhuhasM/RAG-APP.git
cd rag_app

# Create and activate virtual environment
uv venv
.venv\Scripts\activate  # Windows

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI__API_KEY=sk-...
#   PINECONE__API_KEY=pc-...

# Run the application
uv run fastapi dev src/rag_app/app.py
```

**Access Points:**
- Web UI: http://localhost:8000
- Performance Dashboard: http://localhost:8000/dashboard
- API Docs: http://localhost:8000/docs
- Metrics API: http://localhost:8000/api/v1/metrics
- Health Check: http://localhost:8000/api/v1/health

---

## Performance Metrics

### Latency (Production Logs)
```yaml
Query Processing:
  Cold Start (no cache):     3-7 seconds
  Redis Cache Hit:           <500ms
  Semantic Cache Hit:        <800ms
  
Request Flow (10.24s total):
  Query Rewriting:           ~0.5s
  Embedding Generation:      ~1.2s
  Vector Search:             ~0.8s
  Reranking:                 ~0.6s
  LLM Generation:            ~7.0s
  
Resource Initialization:
  Startup Time:              2-5s (one-time reranker load)
  Per-Request Overhead:      0s (singleton pattern)
```

### Observed Metrics from Logs
```yaml
Ingestion:
  Dataset: Complete Works of Mahatma Gandhi (sample)
  Pages Processed: 4
  Nodes Created: 8
  Total Time: 119.39s (~30s/page)
  Batch Size: 100 vectors/batch
  
Query Costs (from logs):
  Input Tokens: 1-50 per query
  Output Tokens: 8-200 per answer
  Cost per Query: $0.0001 - $0.0041 (local model)
  Model Used: gaunernst/gemma-3-1b-it-int4-awq
```

### Optimization Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory per request | 150MB+ | ~50MB | **67% reduction** |
| LLM initialization | ~5-8s | 0s | **100% (cached)** |
| Cache hit latency | N/A | <500ms | **New capability** |
| Semantic cache hit | N/A | <800ms | **New capability** |

*Note: All these metrics are from locally run model llamacpp gemma3-4b on a 6GB VRAM*

---

## Performance Dashboard

The system includes a **real-time performance dashboard** for monitoring and analyzing RAG pipeline performance.

### Access the Dashboard

```bash
# Start the application
uv run fastapi dev src/rag_app/app.py

# Open in browser
http://localhost:8000/dashboard
```

### Dashboard Features

- **Key Metrics Cards**: Total queries, average latency, cache hit rate, cost per query
- **Component Breakdown**: Visual pie chart showing time spent in each pipeline stage
- **Cache Effectiveness**: Bar chart comparing hit rates across cache tiers
- **Provider Distribution**: Doughnut chart showing LLM provider usage
- **Cost Analysis**: Bar chart tracking costs by provider
- **Latency Distribution**: P50, P95, P99 percentile visualization
- **Auto-refresh**: Configurable 30-second updates

### Metrics API

Access raw performance data programmatically:

```bash
# Get comprehensive metrics snapshot
curl http://localhost:8000/api/v1/metrics?limit=100

# Get aggregated statistics
curl http://localhost:8000/api/v1/metrics/aggregated

# Get recent query details
curl http://localhost:8000/api/v1/metrics/recent?limit=10
```

### Run Benchmarks

Execute automated performance benchmarks:

```bash
# Run comprehensive benchmark suite
uv run python scripts/benchmark.py

# View generated performance report
cat PERFORMANCE.md
```

The benchmark script:
- Runs 20+ test queries covering simple to complex scenarios
- Measures end-to-end latency and component breakdowns
- Calculates cache effectiveness and cost metrics
- Generates detailed `PERFORMANCE.md` report

**Example Output:**
```
📊 Metrics Summary:
   - Total Queries: 250
   - Avg Latency: 4523ms
   - Cache Hit Rate: 35.2%
   - Avg Cost: $0.00456
```

---

## Configuration

### Environment Variables

The application uses **Pydantic nested settings** with `__` delimiter for configuration:

```bash
# Required API Keys
OPENAI__API_KEY=sk-proj-...
PINECONE__API_KEY=pc-...
GEMINI__API_KEY=AIzaSy...  # Optional

# Pinecone Vector Database
PINECONE__INDEX_NAME=rag-index
PINECONE__DIMENSION=1024
PINECONE__METRIC=cosine
PINECONE__REGION=us-east-1

# Provider Selection
EMBEDDING__PROVIDER=openai  # openai | gemini | local_models
LLM__PROVIDER=openai        # openai | gemini | local

# Redis Cache
REDIS__HOST=localhost
REDIS__PORT=6379
REDIS__TTL_SECONDS=3600

# RAG Tuning
SEMANTIC_THRESHOLD=0.85
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
CONTEXT_MAX_TOKENS=5000
```

### Cost Tracking

Built-in cost calculation for 15+ models:

```python
# Automatically tracks token usage and calculates costs
# Log example:
# "Request Processed: Input tokens 1, Output tokens 8,
#  Input cost 0.0001, Output cost 0.004, Total cost 0.0041"
```

---
## API Documentation

### Endpoints

#### `POST /api/v1/ingest`
Upload and process PDF documents.

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@documentation.pdf"
```

Response:
```json
{
  "status": "success",
  "documents_processed": 12,
  "chunks_created": 45,
  "vectors_upserted": 45
}
```

---

#### `POST /api/v1/query`
Query the system with SSE streaming.

```bash
curl -N -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain Gandhi'\''s philosophy of non-violence"}'
```

SSE Stream:
```javascript
data: {"type": "status", "data": "Rewriting query..."}
data: {"type": "status", "data": "Retrieving documents..."}
data: {"type": "status", "data": "Generating answer..."}
data: {"type": "token", "data": "Gandhi"}
data: {"type": "token", "data": "'s philosophy"}
data: {"type": "sources", "data": [{"content": "...", "score": 0.92, "source": "page_5"}]}
```

---

#### `GET /api/v1/health`
Component-level health status.

```json
{
  "status": "healthy",
  "timestamp": "2025-10-23T15:27:10Z",
  "components": {
    "vector_db": "operational",
    "llm": "operational",
    "embeddings": "operational",
    "cache": "operational",
    "reranker": "operational"
  },
  "version": "0.1.0"
}
```

---

#### `GET /api/v1/metrics`
Real-time performance metrics.

```bash
curl "http://localhost:8000/api/v1/metrics?limit=100"
```

Response includes:
- Last N queries with detailed breakdowns
- Aggregated statistics (averages, percentiles)
- Cache hit rates across all tiers
- Cost tracking by provider
- Latency distribution (P50, P95, P99)

**Use Cases:**
- Dashboard data source
- Monitoring alerts
- Performance trend analysis
- Cost optimization

---

## Docker Deployment

## Docker Deployment

### Quick Start

```bash
# Configure environment
cp .env.example .env
# Edit .env with API keys

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f rag-app

# Stop
docker-compose down
```

### Features
- Multi-stage build for minimal image size
- Non-root user execution
- Health check integration
- Volume mounts for log persistence
- Hot-reload in development mode (`docker-compose -f docker-compose.dev.yml up`)

### Cloud Deployment

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-app
gcloud run deploy rag-app \
  --image gcr.io/PROJECT_ID/rag-app \
  --platform managed \
  --set-env-vars OPENAI__API_KEY=...,PINECONE__API_KEY=...
```

See [DOCKER.md](./DOCKER.md) for AWS ECS and Azure deployment instructions.

---

## Design Decisions

### Extensibility by Design

The system uses **abstract base classes** and **factory patterns** to support future extensions:

**Current Implementation:**
- PDF-only document support (via PyMuPDF)
- Three LLM providers (OpenAI, Gemini, Local)
- Three embedding providers

**Ready for Extension** (requires minimal code):
- **Document Loaders**: Add `WordDocumentLoader`, `HTMLLoader`, `MarkdownLoader` implementing same interface
- **Vector Stores**: Swap Pinecone for Weaviate, Qdrant, or Milvus via `BaseVectorClient`
- **LLM Providers**: Add Anthropic, Cohere, or custom models implementing `BaseLLMModel`
- **Cache Backends**: Switch Redis for Memcached or DynamoDB via `BaseCacheClient`

**Why Not Implemented?**
- **Scope Management**: Prototype demonstrates architecture without feature bloat
- **Real-World Testing**: PDF support covers 80% of documentation use cases
- **Clean Example**: Focused implementation is easier to understand and extend

### Key Abstractions

```python
# Abstract interfaces enable provider swapping
class BaseLLMModel(ABC):
    @abstractmethod
    async def stream_completion(...) -> AsyncGenerator[str, None]
    
class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_document(document: str) -> List[float]
```

### Trade-offs

| Decision | Why | Alternative Considered |
|----------|-----|------------------------|
| PDF-only ingestion | 80% use case coverage, simpler testing | Multi-format with llama-index loaders |
| Pinecone vector store | Managed service, focus on app logic | Self-hosted Qdrant for cost control |
| Singleton resources | Eliminates cold starts | Per-request init (simpler but slower) |
| LLM for query routing | Adaptive to domain changes | Rule-based classifier (faster but rigid) |
| Semantic cache | Handles paraphrased queries | Exact cache only (simpler but lower hit rate) |

---

## Tech Stack

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI 0.119+ | Async web framework with OpenAPI |
| **Language** | Python 3.12+ | Modern async/await support |
| **LLM Providers** | OpenAI, Gemini, Local (Qwen) | Multi-provider inference |
| **Embeddings** | OpenAI (ada-002), Gemini, Local | Semantic vector generation |
| **Vector DB** | Pinecone | Managed vector search with async |
| **Cache** | Redis (hiredis) | High-performance caching |
| **Document Parser** | PyMuPDF | PDF text extraction |
| **Chunking** | LlamaIndex | Semantic-aware splitting |
| **Reranker** | Sentence-Transformers | CrossEncoder relevance scoring |
| **Tokenizer** | tiktoken | Token counting for context limits |
| **Validation** | Pydantic v2 | Type-safe configuration |
| **Logging** | Loguru | Structured logging |
| **Monitoring** | Custom Performance Tracker | Real-time metrics collection |
| **Deployment** | Docker, Docker Compose | Containerized deployment |
| **Testing** | pytest, pytest-asyncio | Async test framework |
| **Visualization** | Chart.js | Performance dashboard |

---

## Testing

```bash
# Run all tests
uv run pytest -q

# With coverage
uv run pytest --cov=src/rag_app --cov-report=html

# Specific test
uv run pytest tests/test_router_integration.py -v
```

**Coverage:**
- Unit tests for core services
- Integration tests for API routes
- Router classification tests
- Mock-based testing for external services

---

## Documentation

- [Docker Deployment Guide](./DOCKER.md) - Containerization and cloud deployment
- [Architecture Design](./design.md) - System design decisions
- [Performance Metrics](./PERFORMANCE.md) - Detailed benchmarks and optimization roadmap


---

## Author

**Mrudhuhas M**
- Email: mrudhuhas@outlook.com
- GitHub: [@MrudhuhasM](https://github.com/MrudhuhasM)
- LinkedIn: [Connect](https://www.linkedin.com/in/mrudhuhas/)

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- Built following **12-Factor App** methodology
- Implements patterns from production RAG systems
- Leverages best practices from Pinecone, LangChain, and LlamaIndex communities

---

*This project demonstrates production engineering practices in a focused prototype, emphasizing clean architecture and extensibility over feature completeness.*