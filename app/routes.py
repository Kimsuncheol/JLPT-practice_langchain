"""FastAPI routes for health checks, JLPT practice generation, local quiz feedback, tutor chat, and Jinja2 UI."""

import asyncio
import os
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import Settings, enforce_privacy_environment, get_settings
from app.schemas import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ExplanationLanguage,
    HealthResponse,
    JLPTLevel,
    PracticeFollowUpRequest,
    PracticeFollowUpResponse,
    PracticeGenerateRequest,
    PracticeGenerateResponse,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    TutorChatRequest,
    TutorChatResponse,
    TutorRequest,
    TutorResponse,
)
from app.services import (
    OllamaUnavailableError,
    QuotaExceededError,
    StructuralValidationError,
    UnsupportedStructuredOutputError,
    answer_practice_item_tutor_question,
    evaluate_practice_answer,
    generate_follow_up_question,
    generate_practice_batch,
    generate_practice_question,
    get_chat_model,
    is_model_unavailable_error,
    is_quota_or_rate_limit_error,
    is_timeout_error,
    run_tutor_chat,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=[str(TEMPLATES_DIR), str(STATIC_DIR)])

router = APIRouter()


def apply_no_store_headers(response: Response) -> None:
    """Attach mandatory Cache-Control: no-store header to practice and tutor responses."""
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Lightweight health check that never invokes Ollama or external network calls."""
    enforce_privacy_environment()
    tracing_disabled = (
        os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "false"
        and os.environ.get("LANGSMITH_TRACING", "false").lower() == "false"
    )
    return HealthResponse(
        status="ok",
        model=settings.ollama_model,
        base_url_configured=bool(settings.ollama_base_url.strip()),
        tracing_disabled=tracing_disabled,
        persistence="none",
    )


@router.post(
    "/api/practice/generate",
    response_model=PracticeGenerateResponse,
    tags=["practice"],
)
@router.post(
    "/practice/generate",
    response_model=PracticeGenerateResponse,
    tags=["practice"],
    include_in_schema=False,
)
async def generate_practice_questions(
    payload: PracticeGenerateRequest,
    response: Response,
    chat_model: Any = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
) -> PracticeGenerateResponse:
    """Generate unofficial AI-labeled JLPT vocabulary, grammar, or reading questions in one batch call."""
    apply_no_store_headers(response)
    try:
        return await generate_practice_batch(
            payload=payload,
            chat_model=chat_model,
            settings=settings,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except (UnsupportedStructuredOutputError, StructuralValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Practice question generation timed out waiting for Ollama.",
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception as exc:
        if is_timeout_error(exc):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Practice question generation timed out waiting for Ollama.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Ollama endpoint rate limit exceeded. Please wait and try again.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_model_unavailable_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service or selected model is unavailable.",
                headers={"Cache-Control": "no-store"},
            ) from None
        # Never log prompts, answers, or exception details containing user content
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate practice questions from Ollama model.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.post(
    "/api/practice/follow-up",
    response_model=PracticeFollowUpResponse,
    tags=["practice"],
)
@router.post(
    "/practice/follow-up",
    response_model=PracticeFollowUpResponse,
    tags=["practice"],
    include_in_schema=False,
)
async def generate_follow_up_practice_question(
    payload: PracticeFollowUpRequest,
    response: Response,
    chat_model: Any = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
) -> PracticeFollowUpResponse:
    """Generate one original follow-up question targeting the same learning point with Cache-Control: no-store."""
    apply_no_store_headers(response)
    try:
        return await generate_follow_up_question(
            payload=payload,
            chat_model=chat_model,
            settings=settings,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except (UnsupportedStructuredOutputError, StructuralValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Follow-up question generation timed out waiting for Ollama.",
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception as exc:
        if is_timeout_error(exc):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Follow-up question generation timed out waiting for Ollama.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Ollama endpoint rate limit exceeded. Please wait and try again.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_model_unavailable_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service or selected model is unavailable.",
                headers={"Cache-Control": "no-store"},
            ) from None
        # Never log prompts, answers, or exception details containing user content
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate follow-up question from Ollama model.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.post(
    "/practice/question",
    response_model=PracticeQuestionResponse,
    tags=["practice"],
)
@router.post(
    "/api/practice/question",
    response_model=PracticeQuestionResponse,
    tags=["practice"],
)
async def create_practice_question(
    payload: PracticeQuestionRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> PracticeQuestionResponse:
    """Generate a single JLPT practice question with Cache-Control: no-store."""
    apply_no_store_headers(response)
    try:
        return await generate_practice_question(payload, settings=settings)
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception as exc:
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Ollama endpoint rate limit exceeded. Please wait and try again.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_model_unavailable_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service or selected model is unavailable.",
                headers={"Cache-Control": "no-store"},
            ) from None
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate practice question from Ollama model.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.post(
    "/practice/evaluate",
    response_model=AnswerEvaluationResponse,
    tags=["practice"],
)
@router.post(
    "/api/practice/evaluate",
    response_model=AnswerEvaluationResponse,
    tags=["practice"],
)
async def evaluate_answer(
    payload: AnswerEvaluationRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> AnswerEvaluationResponse:
    """Evaluate a submitted JLPT multiple-choice answer locally without calling Ollama."""
    apply_no_store_headers(response)
    return await evaluate_practice_answer(payload, settings=settings)


@router.post(
    "/api/tutor",
    response_model=TutorResponse,
    tags=["tutor"],
)
@router.post(
    "/tutor",
    response_model=TutorResponse,
    tags=["tutor"],
    include_in_schema=False,
)
async def tutor_for_practice_item(
    payload: TutorRequest,
    response: Response,
    chat_model: Any = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
) -> TutorResponse:
    """Answer questions about the selected practice item using a bounded recent conversation window."""
    apply_no_store_headers(response)
    try:
        return await answer_practice_item_tutor_question(
            payload=payload,
            chat_model=chat_model,
            settings=settings,
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except (UnsupportedStructuredOutputError, StructuralValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Tutor request timed out waiting for Ollama.",
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception as exc:
        if is_timeout_error(exc):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Tutor request timed out waiting for Ollama.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Ollama endpoint rate limit exceeded. Please wait and try again.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_model_unavailable_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service or selected model is unavailable.",
                headers={"Cache-Control": "no-store"},
            ) from None
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to process tutor request.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.post(
    "/tutor/chat",
    response_model=TutorChatResponse,
    tags=["tutor"],
)
@router.post(
    "/api/tutor/chat",
    response_model=TutorChatResponse,
    tags=["tutor"],
)
async def tutor_chat(
    payload: TutorChatRequest,
    response: Response,
    chat_model: Any = Depends(get_chat_model),
    settings: Settings = Depends(get_settings),
) -> TutorChatResponse:
    """Stateless JLPT tutor endpoint. Context stays only in browser & request-local memory."""
    apply_no_store_headers(response)
    try:
        return await run_tutor_chat(payload, chat_model=chat_model, settings=settings)
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception as exc:
        if is_timeout_error(exc):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Tutor request timed out waiting for Ollama.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_quota_or_rate_limit_error(exc):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Ollama endpoint rate limit exceeded. Please wait and try again.",
                headers={"Cache-Control": "no-store"},
            ) from None
        if is_model_unavailable_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service or selected model is unavailable.",
                headers={"Cache-Control": "no-store"},
            ) from None
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to process tutor chat request.",
            headers={"Cache-Control": "no-store"},
        ) from None


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the responsive Jinja2 browser interface with ephemeral in-memory study state."""
    context = {
        "app_name": settings.app_name,
        "ollama_model": settings.ollama_model,
        "levels": [lvl.value for lvl in JLPTLevel],
        "categories": [
            {"value": "grammar", "label": "Grammar"},
            {"value": "vocabulary", "label": "Vocabulary"},
            {"value": "reading", "label": "Reading (Passage + up to 3 Qs)"},
        ],
        "languages": [lang.value for lang in ExplanationLanguage],
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )
