from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "ES Query Intelligence - AI Service"
    app_env: str = "development"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    # "anthropic" or "openai" — controls which client gets used in llm_client.py
    # changing this one value switches the entire LLM provider, nothing else changes
    llm_provider: str = "gemini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"

    # kept very low — we want deterministic JSON output, not creative writing
    # higher values = more random = more likely to return malformed JSON
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000

    api_secret_key: str = "AIzaSyB273gXVn9XXyopuzX_TTz8rbHhwCKQpKg"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # env var names are case-insensitive: LLM_PROVIDER and llm_provider both work
        case_sensitive=False,
    )


# @lru_cache means Settings() is only constructed ONCE no matter how many files import it
# without this, every import reads from disk again — wasteful and inconsistent
@lru_cache()
def get_settings() -> Settings:
    return Settings()