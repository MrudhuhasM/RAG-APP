# RAG Prototype Todo List

## Setup and Infrastructure
- [x] Set up project environment with uv (venv, dependencies)
- [x] Configure Pydantic settings for environment-based config
- [x] Set up logging configuration
- [x] Initialize FastAPI app with CORS and routers
- [ ] Create Docker container with all dependencies
- [ ] Set up local development with docker-compose.yml

## Document Ingestion Pipeline
- [x] Implement PDF parser for documentation
- [x] Add intelligent chunking (respect headers, code blocks, tables)
- [x] Integrate embedding generation (OpenAI/Cohere)
- [x] Set up vector database (Pinecone/Weaviate/Qdrant)
- [ ] Implement cloud storage for raw documents
- [ ] Create metadata storage (SQLite/PostgreSQL for doc sets/history)
- [x] Build POST /ingest endpoint for uploading/processing docs

## Query Engine
- [x] Implement semantic search for top-K relevant chunks (K=5)
- [x] Add context window management (max 8K tokens)
- [x] Create prompt template with system instructions for accuracy
- [x] Integrate LLM calls (GPT-4/Claude) for answer generation
- [x] Add source attribution in responses
- [x] Build POST /query endpoint for submitting questions

## API and Web Interface
- [x] Implement GET /sources endpoint for listing doc sets
- [x] Ensure health check endpoint is functional
- [x] Create basic HTML/JavaScript frontend (single file)
- [x] Add error handling for malformed questions/no relevant docs

## Testing and Validation
- [x] Ingest at least 100 pages of documentation without errors
- [ ] Manually verify answers for 8/10 test questions
- [ ] Ensure response time under 10 seconds
- [ ] Test edge cases (multi-part questions, hallucinations)
- [ ] Run unit tests with pytest (70-80% coverage)
- [ ] Perform integration tests with Docker Compose

## Deployment and Finalization
- [ ] Deploy to Cloud Run with public URL
- [ ] Externalize config (API keys in env vars/secrets)
- [ ] Ingest at least 10 real documentation sources
- [ ] Create README with setup instructions
- [ ] Generate test question set with expected answers
- [ ] Create simple architecture diagram
- [ ] Ensure clean git commit history
