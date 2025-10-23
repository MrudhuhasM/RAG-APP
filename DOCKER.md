# Docker Deployment Guide

This guide provides comprehensive instructions for deploying the RAG App using Docker and Docker Compose.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Configuration](#docker-configuration)
- [Deployment Modes](#deployment-modes)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Production Best Practices](#production-best-practices)

## Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher (included with Docker Desktop)
- **API Keys**: OpenAI and Pinecone API keys

### Installing Docker

#### Windows
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Run the installer
3. Start Docker Desktop

#### macOS
1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Install and run Docker Desktop

#### Linux
`ash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
`

## Quick Start

### Using the Helper Script

**Windows (PowerShell):**
`powershell
.\docker-start.ps1
`

**Linux/Mac:**
`ash
chmod +x docker-start.sh
./docker-start.sh
`

### Manual Setup

1. **Copy the environment template:**
   `ash
   cp .env.docker .env
   `

2. **Edit .env with your API keys:**
   `ash
   nano .env  # or use your preferred editor
   `

3. **Build and start the application:**
   `ash
   docker-compose up -d
   `

4. **Access the application:**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health

## Docker Configuration

### File Structure

`
.
 Dockerfile              # Multi-stage production image
 docker-compose.yml      # Production configuration
 docker-compose.dev.yml  # Development configuration with hot-reload
 .dockerignore          # Files excluded from Docker build
 .env.docker            # Environment template
 docker-start.ps1/.sh   # Helper scripts
`

### Dockerfile Details

The Dockerfile uses a **multi-stage build** for optimal image size:

1. **Builder Stage**: Installs dependencies using uv
2. **Runtime Stage**: Copies only necessary files and runs as non-root user

**Key Features:**
- Python 3.12 slim base image
- Non-root user (ppuser) for security
- Health checks for monitoring
- Optimized layer caching
- Minimal attack surface

## Deployment Modes

### Production Mode

Best for: Stable deployments, cloud hosting

`ash
docker-compose up -d
`

**Features:**
- Runs in detached mode
- Automatic restarts on failure
- Production-grade logging
- Resource limits enforced

**Stop the application:**
`ash
docker-compose down
`

### Development Mode

Best for: Local development, debugging

`ash
docker-compose -f docker-compose.dev.yml up
`

**Features:**
- Hot-reload on code changes
- Source code mounted as volumes
- Debug logging enabled
- Console output visible

**Stop with:** Ctrl+C

### Viewing Logs

`ash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View logs for specific service
docker-compose logs -f rag-app

# View last 100 lines
docker-compose logs --tail=100
`

## Environment Variables

### Required Variables

`ash
OPENAI_API_KEY=sk-...           # Your OpenAI API key
PINECONE_API_KEY=...            # Your Pinecone API key
`

### Optional Variables

`ash
# Provider Selection
EMBEDDING_PROVIDER=openai        # Options: openai, gemini, local
LLM_PROVIDER=openai             # Options: openai, gemini, local

# Gemini (if using as provider)
GEMINI_API_KEY=...

# Pinecone Configuration
PINECONE_INDEX_NAME=rag-index
PINECONE_DIMENSION=1536
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
`

### Loading Environment Variables

**From .env file (recommended):**
`ash
# Docker Compose automatically loads .env file
docker-compose up -d
`

**From system environment:**
`ash
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=...
docker-compose up -d
`

**Inline (not recommended for secrets):**
`ash
OPENAI_API_KEY=sk-... PINECONE_API_KEY=... docker-compose up -d
`

## Troubleshooting

### Docker Daemon Not Running

**Error:** Cannot connect to the Docker daemon

**Solution:**
- **Windows/Mac**: Start Docker Desktop
- **Linux**: sudo systemctl start docker

### Port Already in Use

**Error:** Bind for 0.0.0.0:8000 failed: port is already allocated

**Solution:**
`ash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Kill the process or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use external port 8001
`

### Build Failures

**Error:** ailed to solve with frontend dockerfile.v0

**Solutions:**
1. Check Docker version: docker --version (needs 20.10+)
2. Clean build cache: docker builder prune
3. Rebuild without cache: docker-compose build --no-cache

### Container Exits Immediately

**Check logs:**
`ash
docker-compose logs rag-app
`

**Common issues:**
- Missing API keys
- Invalid environment variables
- Syntax errors in configuration files

### Health Check Failing

**Check health status:**
`ash
docker ps
docker inspect rag-app | grep Health -A 10
`

**Common fixes:**
- Wait longer (reranker model download takes time)
- Check if port 8000 is accessible inside container
- Verify API keys are valid

### Out of Memory

**Error:** Container killed due to memory

**Solution:** Increase Docker memory limit
- Docker Desktop: Settings  Resources  Memory (increase to 4GB+)
- Linux: Adjust /etc/docker/daemon.json

## Production Best Practices

### Security

1. **Never commit .env files:**
   `ash
   echo ".env" >> .gitignore
   `

2. **Use secrets management:**
   - Docker Secrets
   - AWS Secrets Manager
   - Google Secret Manager
   - Azure Key Vault

3. **Run as non-root:** Already configured in Dockerfile

4. **Scan for vulnerabilities:**
   `ash
   docker scan rag-app:latest
   `

### Performance

1. **Resource limits:**
   `yaml
   # docker-compose.yml
   services:
     rag-app:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
           reservations:
             memory: 2G
   `

2. **Use multi-stage builds:** Already implemented

3. **Optimize layer caching:** Place frequently changing files last

### Monitoring

1. **Health checks:** Already configured

2. **Logging:**
   `ash
   # Configure JSON logging for production
   docker-compose logs --json > logs.json
   `

3. **Metrics:**
   - CPU/Memory usage: docker stats
   - Container health: docker ps --filter health=healthy

### Backup & Recovery

1. **Volume backups:**
   `ash
   # Backup logs
   docker run --rm -v rag-app_logs:/data -v C:\Users\mrudh\Documents\Projects\ProfileProject\rag_app:/backup \
     alpine tar czf /backup/logs-backup.tar.gz /data
   `

2. **Database backups:** Ensure Pinecone has proper backup strategy

### Cloud Deployment

#### Google Cloud Run

`ash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-app

# Deploy
gcloud run deploy rag-app \
  --image gcr.io/PROJECT_ID/rag-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars OPENAI_API_KEY=\ \
  --set-env-vars PINECONE_API_KEY=\
`

#### AWS ECS

`ash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker tag rag-app:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/rag-app:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/rag-app:latest

# Create ECS task definition and service via AWS Console or CLI
`

#### Azure Container Instances

`ash
# Login to Azure
az login

# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name rag-app \
  --image rag-app:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    OPENAI_API_KEY=\ \
    PINECONE_API_KEY=\
`

## Additional Commands

### Clean Up Everything

`ash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi rag-app:latest

# Remove all unused Docker resources
docker system prune -a
`

### Shell Access

`ash
# Access running container
docker exec -it rag-app bash

# Run one-off command
docker exec rag-app python -c "import sys; print(sys.version)"
`

### Database Connection

`ash
# Test Pinecone connection
docker exec rag-app python -c "
from pinecone import Pinecone
import os
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('Connected:', pc.list_indexes())
"
`

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/your-repo/issues)
- Documentation: See README.md
- API Docs: http://localhost:8000/docs (when running)

---

**Last Updated:** 2025-10-21
