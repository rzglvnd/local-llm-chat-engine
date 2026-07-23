# Deployment Guide

## Container deployment

Build and run:

```bash
docker build -t local-llm-chat-engine .
docker run --rm -p 8001:8001 local-llm-chat-engine
```

Recommended environment variables in production:

- `LOCAL_LLM_API_KEY` (required for write-protected environments)
- `OPENAI_API_KEY` (if using OpenAI backend)
- `LOCAL_LLM_SNAPSHOT_PATH` (point to persistent volume)
- `LOCAL_LLM_LOG_LEVEL=INFO`

## Docker Compose

This repo is wired to the workspace `docker-compose.yml` and can be paired with `erp-ai-assistant`.

## Release process

1. Open PR and pass CI (lint + tests).
2. Tag release from `main`.
3. Deploy image to your registry.
4. Roll out with canary and monitor `/ready`, latency, and error rates.
