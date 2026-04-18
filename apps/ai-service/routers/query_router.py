import structlog
from fastapi import APIRouter, HTTPException

from schemas.analysis import AnalyzeRequest, AnalysisResult
from services.query_analyzer import QueryAnalyzerService

log = structlog.get_logger()

# APIRouter is like a mini Express router — groups related endpoints together
# main.py mounts this with prefix="/api/v1", so POST /analyze becomes POST /api/v1/analyze
router = APIRouter(tags=["Query Analysis"])

# one shared service instance for the lifetime of the app
# we don't create a new one per request — that would be wasteful
analyzer = QueryAnalyzerService()


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    # these show up in Swagger docs — useful for your future self and teammates
    summary="Analyze a natural language query",
    description="Converts a plain English query into an ES DSL query with explanation and suggestions",
)
async def analyze_query(request: AnalyzeRequest) -> AnalysisResult:
    # FastAPI already validated request against AnalyzeRequest schema before we get here
    # if natural_language_query was missing or too short, we never reach this line

    log.info(
        "analyze.request.received",
        query=request.natural_language_query[:80],  # truncate — never log full user input
        has_mapping=request.index_mapping is not None,
    )

    try:
        result = await analyzer.analyze(request)

        log.info(
            "analyze.request.success",
            issues_found=len(result.performance_issues),
            optimizations=len(result.optimizations),
        )

        return result

    except ValueError as e:
        # ValueError means the LLM returned something we couldn't parse into AnalysisResult
        # this is an internal problem, not a bad user request — hence 502 not 400
        log.error("analyze.request.parse_failed", error=str(e))
        raise HTTPException(
            status_code=502,
            detail="AI service returned an unexpected response. Please try again.",
        )

    except Exception as e:
        # catch-all for LLM timeouts, network errors, rate limits etc.
        log.error("analyze.request.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )