"""Tests for JLPT Practice & Tutor FastAPI application, batch generation, and privacy constraints."""

import os
from typing import Any
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.config import Settings, enforce_privacy_environment
from app.main import create_app
from app.schemas import (
    AnswerEvaluationResponse,
    ChatMessage,
    JLPTLevel,
    LLMGeneratedQuestionItem,
    LLMPracticeBatchOutput,
    PracticeCategory,
    PracticeQuestionResponse,
    TutorChatResponse,
    compute_stable_question_id,
)
from app.services import (
    _build_request_local_history,
    create_chat_model,
    get_chat_model,
)


class FakeStructuredChatModel:
    """Fake LangChain-compatible chat model for dependency injection in tests."""

    def __init__(self, batch_output: Any) -> None:
        self.batch_output = batch_output
        self.structured_schema: Any = None
        self.call_count = 0
        self.last_input: Any = None

    def with_structured_output(self, schema: Any) -> RunnableLambda:
        self.structured_schema = schema

        async def _ainvoke(prompt_value: Any) -> Any:
            self.call_count += 1
            self.last_input = prompt_value
            if callable(self.batch_output):
                return self.batch_output(prompt_value)
            return self.batch_output

        return RunnableLambda(_ainvoke)


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


def test_practice_generate_batch_single_call_and_dependency_injection() -> None:
    """POST /api/practice/generate uses injected fake model, makes 1 call per batch, and validates structure."""
    fake_batch = LLMPracticeBatchOutput(
        questions=[
            LLMGeneratedQuestionItem(
                question_text="毎朝、新聞を（　）から会社へ行きます。",
                options=["読んで", "読み", "読む", "読んだ"],
                correct_index=0,
                option_explanations=[
                    "Correct: てから means 'after doing'.",
                    "Incorrect: 読み (masu-stem) does not connect directly to から here.",
                    "Incorrect: Dictionary form + から means 'because', not sequence.",
                    "Incorrect: た形 connects to ら (たら), not から for sequence.",
                ],
            ),
            LLMGeneratedQuestionItem(
                question_text="駅の前に新しい図書館が（　）そうです。",
                options=["できる", "でき", "できて", "できた"],
                correct_index=0,
                option_explanations=[
                    "Correct: Plain form + そうです expresses hearsay.",
                    "Incorrect: Stem + そうです expresses appearance, and でき doesn't take that form.",
                    "Incorrect: て-form cannot precede そうです.",
                    "Incorrect: Here the library is going to be built (future hearsay).",
                ],
            ),
        ]
    )
    fake_model = FakeStructuredChatModel(fake_batch)

    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: fake_model
    client = TestClient(app)

    response = client.post(
        "/api/practice/generate",
        json={
            "level": "N4",
            "category": "grammar",
            "question_count": 2,
            "explanation_language": "English",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "no-store"

    # Verify exactly ONE generation call per batch
    assert fake_model.call_count == 1
    assert fake_model.structured_schema is LLMPracticeBatchOutput

    data = response.json()
    assert data["level"] == "N4"
    assert data["category"] == "grammar"
    assert data["question_count"] == 2
    assert data["explanation_language"] == "English"
    assert data["ai_generated"] is True
    assert "AI-generated" in data["content_label"]

    # Verify structural validation vs linguistic correctness distinction
    assert data["validation"]["structural_validation_passed"] is True
    assert data["validation"]["linguistic_correctness_guaranteed"] is False
    assert "linguistic" in data["validation"]["notice"].lower()

    # Verify each question's structure and stable ID
    assert len(data["questions"]) == 2
    first_q = data["questions"][0]
    expected_id = compute_stable_question_id(
        "N4",
        "grammar",
        "毎朝、新聞を（　）から会社へ行きます。",
        ["読んで", "読み", "読む", "読んだ"],
    )
    assert first_q["id"] == expected_id
    assert first_q["correct_index"] == 0
    assert first_q["correct_answer"] == "読んで"
    assert len(first_q["options"]) == 4
    assert len(first_q["option_explanations"]) == 4
    assert first_q["ai_generated"] is True


def test_practice_generate_supports_korean_and_japanese_and_vocabulary() -> None:
    """POST /api/practice/generate supports N5-N1, vocabulary, and Korean/Japanese explanations."""
    fake_batch = LLMPracticeBatchOutput(
        questions=[
            LLMGeneratedQuestionItem(
                question_text="会議の資料を事前に（　）しておいた。",
                options=["準備", "散歩", "掃除", "洗濯"],
                correct_index=0,
                option_explanations=[
                    "정답: 회의 자료를 사전에 '준비(準備)'해 두었다가 자연스럽습니다.",
                    "오답: 산책(散歩)은 문맥에 맞지 않습니다.",
                    "오답: 청소(掃除)는 자료를 대상으로 쓰지 않습니다.",
                    "오답: 세탁(洗濯)은 옷 등을 빨 때 사용합니다.",
                ],
            )
        ]
    )
    fake_model = FakeStructuredChatModel(fake_batch)

    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: fake_model
    client = TestClient(app)

    response = client.post(
        "/api/practice/generate",
        json={
            "level": "N2",
            "category": "vocabulary",
            "count": 1,
            "explanation_language": "Korean",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["level"] == "N2"
    assert body["category"] == "vocabulary"
    assert body["explanation_language"] == "Korean"
    assert len(body["questions"]) == 1


def test_practice_generate_input_validation_bounds() -> None:
    """Rejects question_count outside 1..5, unsupported categories, and oversized request bodies."""
    fake_model = FakeStructuredChatModel(None)
    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: fake_model
    client = TestClient(app)

    # question_count > 5 rejected (422)
    res_high = client.post(
        "/api/practice/generate",
        json={
            "level": "N5",
            "category": "grammar",
            "question_count": 6,
            "explanation_language": "English",
        },
    )
    assert res_high.status_code == 422
    assert res_high.headers.get("Cache-Control") == "no-store"

    # Unsupported category ('reading') rejected on /api/practice/generate (422)
    res_cat = client.post(
        "/api/practice/generate",
        json={
            "level": "N5",
            "category": "reading",
            "question_count": 1,
            "explanation_language": "English",
        },
    )
    assert res_cat.status_code == 422

    # Oversized body (> 4096 bytes) rejected with 413
    oversized_payload = '{"level":"N5","category":"grammar","padding":"' + ("x" * 5000) + '"}'
    res_large = client.post(
        "/api/practice/generate",
        content=oversized_payload,
        headers={"Content-Type": "application/json"},
    )
    assert res_large.status_code == 413
    assert res_large.headers.get("Cache-Control") == "no-store"
    assert fake_model.call_count == 0


def test_practice_generate_structural_validation_rejects_malformed_model_output() -> None:
    """Structural validation catches duplicate options, non-Japanese text, or mismatched batch count."""
    # Case 1: Model returns duplicate options
    duplicate_options_dict = {
        "questions": [
            {
                "question_text": "これは（　）ですか。",
                "options": ["本", "本", "水", "車"],
                "correct_index": 0,
                "option_explanations": ["A", "B", "C", "D"],
            }
        ]
    }
    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredChatModel(
        duplicate_options_dict
    )
    client = TestClient(app)

    res_dup = client.post(
        "/api/practice/generate",
        json={
            "level": "N5",
            "category": "vocabulary",
            "question_count": 1,
            "explanation_language": "Japanese",
        },
    )
    assert res_dup.status_code == 502
    assert res_dup.headers.get("Cache-Control") == "no-store"
    assert "structural" in res_dup.json()["detail"].lower()

    # Case 2: Model returns non-Japanese question_text
    non_jp_dict = {
        "questions": [
            {
                "question_text": "This question has no Japanese characters at all.",
                "options": ["あ", "い", "う", "え"],
                "correct_index": 0,
                "option_explanations": ["A", "B", "C", "D"],
            }
        ]
    }
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredChatModel(
        non_jp_dict
    )
    res_no_jp = client.post(
        "/api/practice/generate",
        json={
            "level": "N5",
            "category": "vocabulary",
            "question_count": 1,
            "explanation_language": "English",
        },
    )
    assert res_no_jp.status_code == 502

    # Case 3: Model returns 1 question when 2 were requested
    one_question_batch = LLMPracticeBatchOutput(
        questions=[
            LLMGeneratedQuestionItem(
                question_text="日本へ（　）ことがありますか。",
                options=["行った", "行く", "行って", "行き"],
                correct_index=0,
                option_explanations=["1", "2", "3", "4"],
            )
        ]
    )
    app.dependency_overrides[get_chat_model] = lambda: FakeStructuredChatModel(
        one_question_batch
    )
    res_count_mismatch = client.post(
        "/api/practice/generate",
        json={
            "level": "N5",
            "category": "grammar",
            "question_count": 2,
            "explanation_language": "English",
        },
    )
    assert res_count_mismatch.status_code == 502
    assert "requested 2" in res_count_mismatch.json()["detail"]


def test_create_chat_model_passes_bounded_parameters() -> None:
    """create_chat_model passes bounded max_output_tokens, timeout, and max_retries to Gemini."""
    settings = Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key-123",
        GEMINI_MODEL="gemini-3.1-flash-lite",
        GEMINI_MAX_OUTPUT_TOKENS=1536,
        GEMINI_TIMEOUT_SECONDS=20.0,
        GEMINI_MAX_RETRIES=1,
    )
    with patch("app.services.ChatGoogleGenerativeAI") as mock_cls:
        create_chat_model(settings)
        mock_cls.assert_called_once_with(
            model="gemini-3.1-flash-lite",
            google_api_key="test-key-123",
            temperature=0.4,
            max_output_tokens=1536,
            timeout=20.0,
            max_retries=1,
        )


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
    """Calling practice endpoints without GEMINI_API_KEY returns 503 with Cache-Control: no-store."""
    client = TestClient(create_app())
    res = client.post(
        "/api/practice/generate",
        json={
            "level": "N5",
            "category": "vocabulary",
            "question_count": 1,
            "explanation_language": "English",
        },
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
