import json
import structlog
import google.generativeai as genai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from core.config import get_settings
from schemas.analysis import AnalyzeRequest, AnalysisResult

log = structlog.get_logger()
settings = get_settings()


class QueryAnalyzerService:

    def __init__(self):
        # initialize only the client we actually need — no point building all three
        if settings.llm_provider == "anthropic":
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

        elif settings.llm_provider == "openai":
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)

        elif settings.llm_provider == "gemini":
            # Gemini uses a different pattern — configure globally then create model
            genai.configure(api_key=settings.gemini_api_key)
            self._client = genai.GenerativeModel(
                model_name=settings.llm_model,
                # generation_config controls output behavior
                generation_config=genai.GenerationConfig(
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                ),
                # system instruction is Gemini's equivalent of system prompt
                system_instruction="You are an Elasticsearch query expert. You respond only with valid JSON. Never use markdown code fences. Never add explanation outside the JSON.",
            )

        log.info("query_analyzer.initialized", provider=settings.llm_provider)

    async def analyze(self, request: AnalyzeRequest) -> AnalysisResult:
        # three clean steps — each method does exactly one thing
        prompt = self._build_prompt(request)
        raw_response = await self._call_llm(prompt)
        return self._parse_response(raw_response)

    def _build_prompt(self, request: AnalyzeRequest) -> str:
        # tell the LLM exactly what mapping context we have
        # more context = better, more accurate query generation
        mapping_section = (
            f"Index Mapping provided:\n{json.dumps(request.index_mapping, indent=2)}"
            if request.index_mapping
            else "No index mapping provided — infer field types from the query context."
        )

        # the prompt is a strict contract, not a conversation
        # every instruction here directly prevents a class of bad LLM output
        return f"""You are an Elasticsearch expert. Analyze the following natural language query.

{mapping_section}

Natural language query: "{request.natural_language_query}"

You MUST respond with a single valid JSON object. No markdown. No backticks. No explanation outside the JSON.

The JSON must follow this exact schema:
{{
  "es_query": {{...}},
  "explanation": "plain English explanation of how the query works",
  "performance_issues": [
    {{
      "severity": "error" | "warn" | "info",
      "title": "short title",
      "description": "detailed explanation"
    }}
  ],
  "optimizations": [
    {{
      "title": "short title",
      "impact": "high" | "medium" | "low",
      "description": "what this optimization does",
      "optimized_query": {{...}} or null
    }}
  ],
  "mapping_suggestions": [
    {{
      "field": "field_name",
      "current_type": "text" or null,
      "suggested_type": "keyword",
      "reason": "why this change improves performance"
    }}
  ]
}}"""

    async def _call_llm(self, prompt: str) -> str:
        # each provider has a different SDK — we isolate that difference here
        # the rest of the service never needs to know which provider is active
        if settings.llm_provider == "anthropic":
            return await self._call_anthropic(prompt)

        elif settings.llm_provider == "openai":
            return await self._call_openai(prompt)

        elif settings.llm_provider == "gemini":
            return await self._call_gemini(prompt)

        elif settings.llm_provider == "mock":
            return self._call_mock()

        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    async def _call_anthropic(self, prompt: str) -> str:
        response = await self._client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            # system prompt sets the LLM's role — kept separate from user prompt
            system="You are an Elasticsearch query expert. You respond only with valid JSON. Never use markdown code fences.",
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of blocks — we want the first text block
        return response.content[0].text

    async def _call_openai(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are an Elasticsearch query expert. You respond only with valid JSON. Never use markdown code fences.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    async def _call_gemini(self, prompt: str) -> str:
        # Gemini's SDK is not fully async — run in executor to avoid blocking the event loop
        # without this, one slow Gemini call would freeze ALL other requests
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.generate_content(prompt)
        )
        return response.text

    def _call_mock(self) -> str:
        # returns realistic fake data so you can test the full pipeline without any API key
        # swap LLM_PROVIDER=gemini once you have your key
        return json.dumps({
            "es_query": {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"country": "India"}},
                            {"range": {"revenue": {"gt": 1000000}}}
                        ]
                    }
                },
                "sort": [{"revenue": {"order": "desc"}}]
            },
            "explanation": "Filters companies in India with revenue above 1M, sorted by revenue descending.",
            "performance_issues": [
                {
                    "severity": "warn",
                    "title": "Missing keyword mapping",
                    "description": "The country field should be mapped as keyword not text for exact matching."
                }
            ],
            "optimizations": [
                {
                    "title": "Use filter context",
                    "impact": "high",
                    "description": "Queries in filter context skip scoring — faster and cacheable.",
                    "optimized_query": None
                }
            ],
            "mapping_suggestions": [
                {
                    "field": "country",
                    "current_type": None,
                    "suggested_type": "keyword",
                    "reason": "Exact match fields should be keyword type to avoid unnecessary tokenization."
                }
            ]
        })

    def _parse_response(self, raw: str) -> AnalysisResult:
        try:
            # LLMs sometimes wrap JSON in ```json ... ``` even when told not to
            # strip that defensively before parsing
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]

            data = json.loads(cleaned)

            # Pydantic validates every field and nested model here
            # if the LLM missed a required field or used wrong types — this raises ValueError
            return AnalysisResult(**data)

        except json.JSONDecodeError as e:
            # LLM returned something that isn't JSON at all — not even parseable
            log.error("parse.json_failed", error=str(e), raw=raw[:200])
            raise ValueError(f"LLM response was not valid JSON: {e}")

        except Exception as e:
            # JSON parsed fine but didn't match our AnalysisResult schema
            log.error("parse.validation_failed", error=str(e))
            raise ValueError(f"LLM response did not match expected schema: {e}")