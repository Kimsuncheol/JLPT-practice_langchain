"""FastAPI entrypoint for the privacy-first JLPT Practice & Tutor application."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from app.config import enforce_privacy_environment, get_settings
from app.routes import router
from app.services import MissingCredentialsError


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

    @application.exception_handler(MissingCredentialsError)
    async def missing_credentials_handler(
        _: Request, exc: MissingCredentialsError
    ) -> JSONResponse:
        """Handle missing backend Gemini credentials with 503 and Cache-Control: no-store."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @application.middleware("http")
    async def privacy_and_bounds_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Enforce bounded input size and guarantee Cache-Control: no-store on practice/tutor routes."""
        path = request.url.path
        is_sensitive_route = "/practice" in path or "/tutor" in path

        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > settings.max_request_body_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={
                        "detail": (
                            f"Request body exceeds maximum allowed size of "
                            f"{settings.max_request_body_bytes} bytes."
                        )
                    },
                    headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
                )

        response = await call_next(request)
        if is_sensitive_route:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
        return response

    application.include_router(router)
    return application


app = create_app()
