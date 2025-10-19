# RAG Prototype Design Document

## Overview
This document outlines the design for a Retrieval-Augmented Generation (RAG) system that ingests technical documentation and answers user questions using Large Language Models (LLMs). The system focuses on correctness and accuracy for technical queries.

## Architecture

### 3-Tier Architecture
`
        
   Web Interface          API Layer         Data Layer     
                                                           
 - HTML/JS UI     - FastAPI         - Vector DB     
 - User Queries       - Pydantic            - Cloud Storage 
 - Doc Selection      - Error Handling      - Metadata DB   
-        
`

### Components

#### 1. Document Ingestion Pipeline
- **Input**: Markdown/HTML documentation files
- **Processing**:
  - Parse documents using markdown/HTML parsers
  - Intelligent chunking (preserves headers, code blocks, tables)
  - Generate embeddings using OpenAI/Cohere APIs
  - Store embeddings in vector database
  - Store raw documents in cloud storage
  - Update metadata in relational database

#### 2. Query Engine
- **Input**: User questions
- **Processing**:
  - Semantic search in vector database (top-K=5 chunks)
  - Context window management (max 8K tokens)
  - Construct prompts with system instructions
  - Call LLM (GPT-4/Claude) for answer generation
  - Include source attribution

#### 3. Web Interface
- Simple HTML/JavaScript single-page application
- Document set selection
- Question input and answer display
- Source attribution display

## Data Flow

1. **Ingestion Flow**:
   User uploads docs  Parse  Chunk  Embed  Store vectors  Store metadata

2. **Query Flow**:
   User asks question  Search vectors  Retrieve chunks  Build prompt  Call LLM  Return answer with sources

## API Design

### Endpoints
- POST /api/v1/ingest - Upload and process documentation
- POST /api/v1/query - Submit question, get answer
- GET /api/v1/sources - List available documentation sets
- GET /api/v1/health - Health check

### Request/Response Models (Pydantic)

#### Ingest Request
`python
class IngestRequest(BaseModel):
    source_url: str
    doc_set_name: str
    content: str  # or file upload
`

#### Query Request
`python
class QueryRequest(BaseModel):
    question: str
    doc_set_name: str
    max_tokens: Optional[int] = 8000
`

#### Query Response
`python
class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    processing_time: float

class Source(BaseModel):
    doc_set: str
    chunk_id: str
    relevance_score: float
    text_preview: str
`

## Storage Design

### Vector Database
- **Purpose**: Store document embeddings for semantic search
- **Options**: Pinecone, Weaviate, Qdrant (cloud managed)
- **Schema**:
  - Vector: 1536 dimensions (OpenAI ada-002)
  - Metadata: chunk_id, doc_set, source_url, text_content

### Cloud Storage
- **Purpose**: Store raw documentation files
- **Options**: Google Cloud Storage, AWS S3
- **Structure**: /doc-sets/{name}/{filename}

### Metadata Database
- **Purpose**: Track document sets and ingestion history
- **Options**: SQLite (dev), PostgreSQL (prod)
- **Tables**:
  - doc_sets: id, name, description, created_at
  - ingestions: id, doc_set_id, source_url, status, created_at
  - chunks: id, doc_set_id, vector_id, content, metadata

## Configuration

### Environment Variables
- OPENAI_API_KEY: For embeddings and LLM calls
- PINECONE_API_KEY: Vector database access
- PINECONE_ENVIRONMENT: Vector database environment
- PINECONE_INDEX_NAME: Vector database index
- DATABASE_URL: Metadata database connection
- GCS_BUCKET_NAME: Cloud storage bucket
- APP_ENV: dev/prod environment

### Settings (Pydantic)
`python
class Settings(BaseSettings):
    app_name: str = 'RAG App'
    app_version: str = '1.0.0'
    openai_api_key: str
    pinecone_api_key: str
    # ... other configs
`

## Deployment

### Local Development
- Docker container with all dependencies
- docker-compose.yml for local services (vector DB, metadata DB)
- Hot reload for FastAPI app

### Production
- Google Cloud Run service
- Cloud SQL for metadata
- Cloud Storage for documents
- Pinecone for vectors
- Secret Manager for API keys

### Docker Configuration
`dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD [\"uvicorn\", \"src.rag_app.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
`

## Security Considerations
- API key management via environment variables/secrets
- Input validation and sanitization
- Rate limiting for API endpoints
- CORS configuration for web interface
- No sensitive data in logs

## Monitoring and Logging
- Structured logging with request IDs
- Health check endpoint
- Basic metrics (request count, latency, error rate)
- Log aggregation for production

## Performance Targets
- Ingestion: Process 100 pages without errors
- Query: Response under 10 seconds
- Accuracy: 8/10 test questions correct
- Context window: Max 8K tokens

## Future Extensions
- Multi-modal support (images, diagrams)
- Advanced chunking strategies
- Fine-tuned models for specific domains
- Multi-tenant architecture
- Caching layer for frequent queries
