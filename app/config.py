"""Application configuration and strict privacy enforcement for Ollama backend."""

import os
from functools import lru_cache
from urllib.parse import urlparse
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Enforce mandatory privacy rules before any LangChain modules are imported or used.
# Disable LangChain and LangSmith tracing unconditionally.
TRACING_ENV_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
)
PROHIBITED_CREDENTIAL_ENV_VARS = (
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_PROJECT",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)

MAX_REQUEST_BODY_BYTES = 4096
LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def enforce_privacy_environment() -> None:
    """Force tracing off and strip any LangSmith or third-party cloud API keys from environment."""
    for var in TRACING_ENV_VARS:
        os.environ[var] = "false"
    for var in PROHIBITED_CREDENTIAL_ENV_VARS:
        os.environ.pop(var, None)


enforce_privacy_environment()


def is_localhost_url(url: str) -> bool:
    """Return True if the URL points to a loopback / localhost address."""
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        return host in LOCALHOST_HOSTS
    except Exception:
        return False


class Settings(BaseSettings):
    """Backend configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="JLPT Practice & Tutor API", alias="APP_NAME")
    ollama_base_url: str = Field(
        default="https://ollama.com",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "ollama_base_url"),
    )
    ollama_model: str = Field(
        default="gpt-oss:120b",
        validation_alias=AliasChoices("OLLAMA_MODEL", "ollama_model"),
    )
    ollama_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "OLLAMA_API_KEY",
            "OLLAMA_AUTH_TOKEN",
            "ollama_api_key",
        ),
    )
    ollama_temperature: float = Field(
        default=0.4, ge=0.0, le=2.0, alias="OLLAMA_TEMPERATURE"
    )
    ollama_num_predict: int = Field(
        default=2048, ge=256, le=4096, alias="OLLAMA_NUM_PREDICT"
    )
    ollama_timeout_seconds: float = Field(
        default=45.0, ge=1.0, le=120.0, alias="OLLAMA_TIMEOUT_SECONDS"
    )
    max_request_body_bytes: int = Field(
        default=MAX_REQUEST_BODY_BYTES, ge=512, le=16384, alias="MAX_REQUEST_BODY_BYTES"
    )
    render_environment: bool = Field(default=False, alias="RENDER")

    def model_post_init(self, __context: object) -> None:
        """Re-enforce tracing disabled even if .env attempted to enable it."""
        enforce_privacy_environment()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings without making external network calls."""
    enforce_privacy_environment()
    return Settings()
