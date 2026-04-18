import structlog
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import query_router
from core.config import get_settings

log = structlog.get_logger()

# called once here at module load — same cached object shared across the whole app
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # code BEFORE yield runs on startup — good place for DB connections, LLM client warmup
    log.info("service.startup", env=settings.app_env, provider=settings.llm_provider)
    yield
    # code AFTER yield runs on shutdown — good place for cleanup, closing connections
    log.info("service.shutdown")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    # Swagger UI only exposed in dev — you don't want strangers exploring your API in prod
    docs_url="/docs" if settings.debug else None,
    lifespan=lifespan,
)

# allows NestJS (localhost:3000) to call this service during local development
# in production: replace with your real API domain, never use "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# mounts all routes from query_router under /api/v1
# so POST /analyze in the router becomes POST /api/v1/analyze
app.include_router(query_router.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    # NestJS and Docker will ping this to confirm the service is alive
    return {
        "status": "ok",
        "env": settings.app_env,
        "provider": settings.llm_provider,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        # reload=True watches files and restarts on save, like nodemon — dev only
        reload=settings.debug,
    )