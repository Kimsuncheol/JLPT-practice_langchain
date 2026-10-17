"""Application configuration and strict privacy enforcement."""

import os
from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Enforce mandatory privacy rules before any LangChain modules are imported or used.
# Disable LangChain and LangSmith tracing unconditionally.
TRACING_ENV_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
)
LANGSMITH_CREDENTIAL_ENV_VARS = (
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_PROJECT",
)

MAX_REQUEST_BODY_BYTES = 4096


def enforce_privacy_environment() -> None:
    """Force tracing off and strip any LangSmith configuration from environment."""
    for var in TRACING_ENV_VARS:
        os.environ[var] = "false"
    for var in LANGSMITH_CREDENTIAL_ENV_VARS:
        os.environ.pop(var, None)


enforce_privacy_environment()


class Settings(BaseSettings):
    """Backend configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="JLPT Practice & Tutor API", alias="APP_NAME")
    gemini_api_key: SecretStr = Field(default=SecretStr(""), alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(
        default=0.4, ge=0.0, le=2.0, alias="GEMINI_TEMPERATURE"
    )
    gemini_max_output_tokens: int = Field(
        default=2048, ge=256, le=4096, alias="GEMINI_MAX_OUTPUT_TOKENS"
    )
    gemini_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=60.0, alias="GEMINI_TIMEOUT_SECONDS"
    )
    gemini_max_retries: int = Field(
        default=2, ge=0, le=3, alias="GEMINI_MAX_RETRIES"
    )
    max_request_body_bytes: int = Field(
        default=MAX_REQUEST_BODY_BYTES, ge=512, le=16384, alias="MAX_REQUEST_BODY_BYTES"
    )

    def model_post_init(self, __context: object) -> None:
        """Re-enforce tracing disabled even if .env attempted to enable it."""
        enforce_privacy_environment()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings without making external API calls."""
    enforce_privacy_environment()
    return Settings()
