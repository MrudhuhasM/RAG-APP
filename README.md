# RAG App

A Retrieval-Augmented Generation (RAG) system designed to ingest technical documentation, process user questions, and return accurate answers using Large Language Models (LLMs). This project focuses on building a functional prototype with document ingestion, query processing, and a basic web interface.

## Features

- **Document Ingestion Pipeline**: Parses markdown/HTML documentation, chunks it intelligently, generates embeddings, and stores in a vector database.
- **Query Engine**: Accepts user questions, retrieves relevant chunks, constructs prompts, and calls LLMs for answer generation.
- **Vector Storage**: Uses Pinecone for cloud-managed vector storage.
- **Embeddings**: Supports OpenAI and Gemini embeddings.
- **Local Inference**: Integrates with LlamaCPP for local model inference.
- **API Endpoints**: RESTful API with endpoints for ingestion, querying, and health checks.
- **Web Interface**: Simple UI for selecting documentation sets and asking questions.

## Architecture

The application is built as a FastAPI service with the following components:

- **Ingestion Service**: Handles document parsing, chunking, and embedding generation.
- **Vector Client**: Manages interactions with Pinecone vector database.
- **Embedding Clients**: Interfaces for OpenAI and Gemini embedding APIs.
- **API Routes**: Endpoints for ingestion, querying, and system health.

## Performance Benchmarks

### Ingestion Pipeline (Initial Completion)
- **Processed**: 4 pages, 4 documents
- **Output**: Divided into 8 nodes and upserted to vector DB
- **Time Taken**: 119.39 seconds
- **Inference**: Local inference using LlamaCPP

## Installation

### Prerequisites
- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag_app
   ```

2. Create and activate virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   # or
   uv sync
   ```

4. Configure environment variables (see Configuration section).

5. Run the application:
   ```bash
   uv run fastapi dev src/rag_app/app.py
   ```

## Configuration

The application uses Pydantic Settings for configuration. Set the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment
- `PINECONE_INDEX_NAME`: Pinecone index name
- `PINECONE_DIMENSION`: Embedding dimension
- `PINECONE_METRIC`: Distance metric
- `PINECONE_CLOUD`: Cloud provider
- `PINECONE_REGION`: Region
- `LOCAL_MODELS_COMPLETION_BASE_URL`: Base URL for local models (e.g., LlamaCPP)

Create a `.env` file in the project root with these variables.

## Usage

### API Endpoints
- `POST /ingest`: Upload and process new documentation
- `POST /query`: Submit a question and get an answer
- `GET /sources`: List available documentation sets
- `GET /health`: Health check

### Example Query
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Kubernetes?", "source": "kubernetes-docs"}'
```

## Testing

Run tests with:
```bash
uv run pytest -q
```

## Deployment

The application is containerized and can be deployed to Cloud Run or similar platforms.

Build and run locally with Docker:
```bash
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

## Contributing

Follow the production principles outlined in the project instructions. Ensure code quality with linting, formatting, and testing.

## License

[Specify license if applicable]