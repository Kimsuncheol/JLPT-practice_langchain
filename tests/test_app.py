"""Tests for JLPT Practice & Tutor FastAPI application and privacy constraints."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.config import Settings, enforce_privacy_environment
from app.main import create_app
from app.schemas import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ChatMessage,
    JLPTLevel,
    PracticeCategory,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    TutorChatRequest,
    TutorChatResponse,
)
from app.services import (
     _build_request_local_history,
    run_tutor_chat,
)


def test_health_does_not_call_gemini() -> None:
    """GET /health must return 200 OK with configured model without calling Gemini."""
    with patch("app.services.ChatGoogleGenerativeAI") as mock_genai:
        client = TestClient(create_app())
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model"] == "gemini-3.1-flash-lite"
        assert body["tracing_disabled"] is True
        assert body["persistence"] == "none"
        mock_genai.assert_not_called()


def test_langsmith_and_tracing_strictly_disabled(monkeypatch) -> None:
    """Tracing environment variables must be forced to false and LangSmith keys removed."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_secret")

    enforce_privacy_environment()

    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
    assert os.environ.get("LANGSMITH_TRACING") == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_practice_and_tutor_responses_include_no_store_header() -> None:
    """Practice and tutor endpoints must return Cache-Control: no-store."""
    client = TestClient(create_app())

    mock_question = PracticeQuestionResponse(
        level=JLPTLevel.N5,
        category=PracticeCategory.GRAMMAR,
        question="わたし（　）がくせいです。",
        options=["は", "を", "に", "で"],
        correct_index=0,
        explanation="は (wa) marks the topic of the sentence.",
    )
    with patch(
        "app.routes.generate_practice_question",
        return_value=mock_question,
    ):
        res_q = client.post(
            "/api/practice/question",
            json={"level": "N5", "category": "grammar", "topic": "particles"},
        )
        assert res_q.status_code == 200
        assert res_q.headers.get("Cache-Control") == "no-store"
        assert res_q.json()["correct_index"] == 0

    mock_eval = AnswerEvaluationResponse(
        is_correct=True,
        feedback="Correct! は marks the topic.",
        grammar_notes=["N1 は N2 です"],
    )
    with patch(
        "app.routes.evaluate_practice_answer",
        return_value=mock_eval,
    ):
        res_e = client.post(
            "/api/practice/evaluate",
            json={
                "level": "N5",
                "question": "わたし（　）がくせいです。",
                "selected_answer": "は",
            },
        )
        assert res_e.status_code == 200
        assert res_e.headers.get("Cache-Control") == "no-store"
        assert res_e.json()["is_correct"] is True

    mock_tutor = TutorChatResponse(
        level=JLPTLevel.N5,
        reply="「は」is pronounced 'wa' when used as a topic particle.",
    )
    with patch(
        "app.routes.run_tutor_chat",
        return_value=mock_tutor,
    ):
        res_t = client.post(
            "/api/tutor/chat",
            json={
                "level": "N5",
                "message": "How do I use は?",
                "history": [{"role": "user", "content": "Hello"}],
            },
        )
        assert res_t.status_code == 200
        assert res_t.headers.get("Cache-Control") == "no-store"
        assert "wa" in res_t.json()["reply"]


def test_missing_credentials_returns_503_with_no_store() -> None:
    """Calling practice or tutor without GEMINI_API_KEY returns 503 with Cache-Control: no-store."""
    client = TestClient(create_app())
    res = client.post(
        "/api/practice/question",
        json={"level": "N5", "category": "vocabulary"},
    )
    assert res.status_code == 503
    assert res.headers.get("Cache-Control") == "no-store"


def test_request_local_history_conversion() -> None:
    """Conversation history conversion creates a fresh request-local list."""
    history = [
        ChatMessage(role="user", content="こんにちは"),
        ChatMessage(role="assistant", content="こんにちは！何か質問はありますか？"),
    ]
    converted = _build_request_local_history(history)
    assert len(converted) == 2
    assert converted[0].content == "こんにちは"
    assert converted[1].content == "こんにちは！何か質問はありますか？"


def test_configurable_gemini_model_default() -> None:
    """Default GEMINI_MODEL must be gemini-3.1-flash-lite and configurable."""
    default_settings = Settings(_env_file=None)
    assert default_settings.gemini_model == "gemini-3.1-flash-lite"

    custom_settings = Settings(_env_file=None, GEMINI_MODEL="gemini-2.5-flash")
    assert custom_settings.gemini_model == "gemini-2.5-flash"
