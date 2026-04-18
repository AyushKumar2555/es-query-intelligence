# ES Query Intelligence Platform

AI-powered platform that converts natural language into Elasticsearch DSL queries.

## What it does
- Natural language → Elasticsearch DSL query
- Explains how the query works in plain English
- Detects performance issues automatically
- Suggests query optimizations
- Analyzes index mappings and suggests improvements

## Architecture
Frontend → NestJS API → FastAPI AI Service → LLM (Gemini)
↓
PostgreSQL
## Services
- `apps/ai-service/` — Python FastAPI + Gemini LLM
- `apps/api/` — NestJS API Gateway (coming soon)

## Quick Start

### AI Service
```bash
cd apps/ai-service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add your Gemini API key to .env
python main.py
```

### Test
```bash
pytest tests/ -v
```

### API

POST http://localhost:8000/api/v1/analyze
{
"natural_language_query": "Find companies in India with revenue greater than 1 million",
"index_mapping": { ... }
}

## Stack
- Python 3.11+
- FastAPI
- Pydantic v2
- Google Gemini
- NestJS (coming soon)
- PostgreSQL (coming soon)