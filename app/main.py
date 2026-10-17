"""FastAPI entrypoint for the privacy-first JLPT Practice & Tutor application."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response

from app.config import enforce_privacy_environment, get_settings
from app.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Application lifespan hook.

    Enforces privacy environment settings on startup without making any
    network or Gemini API calls.
    """
    enforce_privacy_environment()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    enforce_privacy_environment()
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Stateless JLPT practice and tutor API powered by FastAPI, Pydantic, "
            "and langchain-google-genai with strict privacy guarantees."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def no_store_privacy_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Guarantee Cache-Control: no-store on all practice and tutor endpoints."""
        response = await call_next(request)
        path = request.url.path
        if "/practice" in path or "/tutor" in path:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response

    application.include_router(router)
    return application


app = create_app()
