# Docker Deployment Guide

## Quick Start

### Build the Docker image:
```bash
docker build -t onboarding-assistant:latest .
```

### Run the container:
```bash
docker run --rm onboarding-assistant:latest
```

### Using Docker Compose:
```bash
docker-compose up
```

## Container Details

- **Base Image**: python:3.9-slim
- **Working Directory**: /app
- **Exposed Port**: 8000 (for future API)
- **Health Check**: Enabled (30s interval)

## Volume Mounts

The container uses the following volumes:
- `./data` - Input data
- `./models` - Model artifacts and vector store
- `./experiments` - MLflow tracking and results

## Environment Variables

- `PYTHONUNBUFFERED=1` - Real-time logging
- `PYTHONDONTWRITEBYTECODE=1` - No .pyc files
- `PYTHONPATH=/app` - Module imports

## Production Deployment

For production, consider:
1. Using multi-stage builds to reduce image size
2. Implementing proper secrets management
3. Setting up persistent volumes for models
4. Configuring resource limits
5. Implementing proper logging and monitoring

## Troubleshooting

**Issue**: Module not found
**Solution**: Ensure PYTHONPATH is set correctly

**Issue**: Permission denied
**Solution**: Check volume mount permissions

**Issue**: Out of memory
**Solution**: Increase Docker memory allocation