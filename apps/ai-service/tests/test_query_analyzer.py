import pytest
import json
from services.query_analyzer import QueryAnalyzerService
from schemas.analysis import AnalyzeRequest, AnalysisResult


# ─── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def service():
    # creates one fresh service instance per test
    return QueryAnalyzerService()

@pytest.fixture
def basic_request():
    # the simplest valid request — used across multiple tests
    return AnalyzeRequest(
        natural_language_query="Find companies in India with revenue greater than 1 million"
    )

@pytest.fixture
def request_with_mapping():
    # request with an index mapping — tests the richer prompt path
    return AnalyzeRequest(
        natural_language_query="Find companies in India with revenue greater than 1 million",
        index_mapping={
            "mappings": {
                "properties": {
                    "country":  {"type": "keyword"},
                    "revenue":  {"type": "float"},
                    "name":     {"type": "text"}
                }
            }
        }
    )


# ─── test 1: prompt builder ───────────────────────────────────────────────────

class TestBuildPrompt:

    def test_prompt_contains_user_query(self, service, basic_request):
        prompt = service._build_prompt(basic_request)

        # the user's query must appear in the prompt — if not, the LLM has no idea what to do
        assert basic_request.natural_language_query in prompt

    def test_prompt_contains_json_schema(self, service, basic_request):
        prompt = service._build_prompt(basic_request)

        # these keys must be in the prompt so the LLM knows the exact shape to return
        assert "es_query" in prompt
        assert "performance_issues" in prompt
        assert "optimizations" in prompt
        assert "mapping_suggestions" in prompt

    def test_prompt_without_mapping_says_no_mapping(self, service, basic_request):
        prompt = service._build_prompt(basic_request)

        # when no mapping is provided, the prompt should tell the LLM to infer types
        assert "No index mapping provided" in prompt

    def test_prompt_with_mapping_includes_mapping_json(self, service, request_with_mapping):
        prompt = service._build_prompt(request_with_mapping)

        # when mapping IS provided, it should appear in the prompt for context
        assert "keyword" in prompt
        assert "country" in prompt


# ─── test 2: response parser ──────────────────────────────────────────────────

class TestParseResponse:

    def test_parses_valid_response(self, service):
        # a perfectly valid LLM response — should parse cleanly into AnalysisResult
        valid_json = json.dumps({
            "es_query": {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"country": "India"}},
                            {"range": {"revenue": {"gt": 1000000}}}
                        ]
                    }
                }
            },
            "explanation": "Filters companies in India with revenue above 1M.",
            "performance_issues": [
                {
                    "severity": "warn",
                    "title": "Missing keyword mapping",
                    "description": "country field should be keyword type."
                }
            ],
            "optimizations": [
                {
                    "title": "Use filter context",
                    "impact": "high",
                    "description": "Filter context skips scoring — faster.",
                    "optimized_query": None
                }
            ],
            "mapping_suggestions": [
                {
                    "field": "country",
                    "current_type": None,
                    "suggested_type": "keyword",
                    "reason": "Exact match fields should be keyword."
                }
            ]
        })

        result = service._parse_response(valid_json)

        # check the return type first
        assert isinstance(result, AnalysisResult)
        # check key fields made it through correctly
        assert "query" in result.es_query
        assert result.explanation != ""
        assert len(result.performance_issues) == 1
        assert result.performance_issues[0].severity.value == "warn"

    def test_parses_response_wrapped_in_markdown(self, service):
        # LLMs often wrap JSON in ```json ... ``` even when told not to
        # our parser should strip this and still succeed
        wrapped = "```json\n" + json.dumps({
            "es_query": {"query": {"match_all": {}}},
            "explanation": "Returns all documents.",
            "performance_issues": [],
            "optimizations": [],
            "mapping_suggestions": []
        }) + "\n```"

        # should not raise — stripping logic must handle this
        result = service._parse_response(wrapped)
        assert isinstance(result, AnalysisResult)

    def test_raises_value_error_on_invalid_json(self, service):
        # completely broken response — not JSON at all
        with pytest.raises(ValueError, match="not valid JSON"):
            service._parse_response("Sorry, I cannot help with that.")

    def test_raises_value_error_on_missing_required_field(self, service):
        # valid JSON but missing es_query — Pydantic should reject this
        incomplete = json.dumps({
            "explanation": "Some explanation",
            "performance_issues": [],
            "optimizations": [],
            "mapping_suggestions": []
            # es_query is missing — AnalysisResult requires it
        })

        with pytest.raises(ValueError):
            service._parse_response(incomplete)

    def test_raises_value_error_on_wrong_severity_value(self, service):
        # severity must be "error", "warn", or "info" — anything else should fail
        bad_severity = json.dumps({
            "es_query": {"query": {"match_all": {}}},
            "explanation": "Test.",
            "performance_issues": [
                {
                    "severity": "critical",  # not a valid IssueSeverity value
                    "title": "Some issue",
                    "description": "Some description"
                }
            ],
            "optimizations": [],
            "mapping_suggestions": []
        })

        with pytest.raises(ValueError):
            service._parse_response(bad_severity)

    def test_empty_lists_are_valid(self, service):
        # when the LLM finds no issues and has no suggestions — empty lists are fine
        minimal = json.dumps({
            "es_query": {"query": {"match_all": {}}},
            "explanation": "Returns all documents.",
            "performance_issues": [],
            "optimizations": [],
            "mapping_suggestions": []
        })

        result = service._parse_response(minimal)
        assert result.performance_issues == []
        assert result.optimizations == []


# ─── test 3: full analyze() with mock ────────────────────────────────────────

class TestAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_returns_analysis_result(self, service, basic_request):
        # with LLM_PROVIDER=mock this calls _call_mock() — no real API needed
        result = await service.analyze(basic_request)

        assert isinstance(result, AnalysisResult)
        assert result.es_query is not None
        assert result.explanation != ""

    @pytest.mark.asyncio
    async def test_analyze_with_mapping_returns_result(self, service, request_with_mapping):
        result = await service.analyze(request_with_mapping)

        assert isinstance(result, AnalysisResult)