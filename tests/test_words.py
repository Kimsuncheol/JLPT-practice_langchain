"""Tests for POST /api/words/explain and the Word Explanation UI using a fake model (no running Ollama)."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.main import create_app
from app.services import get_chat_model
from app.words import (
    CuratedExamSourceRecord,
    get_exam_source_collection,
    load_curated_exam_sources,
)

DETAILED_CONTENT: dict[str, Any] = {
    "reading": "じゅんび",
    "is_known_word": True,
    "alternate_readings": [],
    "uncertainty_note": None,
    "context_fit_explanation": "In the provided meeting context, 準備 refers to preparing materials in advance.",
    "meanings": [
        {
            "meaning": "Preparation; arrangements; getting ready",
            "part_of_speech": "Noun / する verb ( transitive )",
            "context_fit": "Directly fits preparing meeting documents beforehand.",
            "collocations": ["準備をする", "心の準備", "準備が整う"],
            "grammar_patterns": ["〜の準備をする", "〜を準備しておく"],
            "examples": [
                {
                    "japanese": "旅行の準備はもう終わりましたか。",
                    "reading": "りょこうのじゅんびはもうおわりましたか。",
                    "translation": "Have you finished preparing for the trip already?",
                    "usage_note": "Used with の to modify 準備 as a noun.",
                }
            ],
        }
    ],
    "nuances": [
        "準備 focuses on concrete, practical preparations before an event or task.",
        "Compared with 用意 (ようい), 準備 often implies broader multi-step readiness.",
    ],
    "register_and_formality": "Neutral register suitable for both everyday conversation and formal business contexts.",
    "synonyms_and_contrasts": [
        {
            "term": "用意 (ようい)",
            "comparison": "用意 often focuses on having specific items ready to use immediately, whereas 準備 covers broader advance preparation.",
        },
        {
            "term": "支度 (したく)",
            "comparison": "支度 is used mainly for personal/household readiness (getting dressed, preparing a meal).",
        },
    ],
    "natural_vs_unnatural": [
        {
            "natural_example": "会議の資料を準備しておいた。",
            "unnatural_example": "会議の資料を身支度しておいた。",
            "explanation": "身支度 is only for personal grooming/dressing, not business documents.",
        }
    ],
    "follow_up_questions": [
        "準備와 用意는 어떻게 다른가요?",
        "準備를 사용한 다른 예문을 알려 주세요.",
        "「準備中」는 어떤 상황에서 쓰나요?",
    ],
}

MINIMAL_CONTENT: dict[str, Any] = {
    "reading": "やさしい",
    "meanings": [
        {
            "meaning": "Kind; considerate",
            "part_of_speech": "い-adjective",
            "examples": [
                {
                    "japanese": "彼は優しいです。",
                    "reading": "かれはやさしいです。",
                    "translation": "He is kind.",
                    "usage_note": "Describes character.",
                }
            ],
        }
    ],
    "nuances": [
        "優しい describes kindness; 易しい describes ease despite the same reading."
    ],
    "follow_up_questions": [
        "How is 優しい different from 親切?",
        "Can you give another example of 優しい?",
        "Why is 易しい written differently?",
    ],
}


def client_for(
    output: Any = DETAILED_CONTENT,
    error: BaseException | None = None,
    sources: list[CuratedExamSourceRecord] | None = None,
) -> tuple[TestClient, list[Any]]:
    calls: list[Any] = []

    async def model(prompt: Any) -> AIMessage:
        calls.append(prompt.to_messages())
        if error:
            raise error
        payload_str = (
            output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        )
        return AIMessage(content=payload_str)

    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: RunnableLambda(model)
    if sources is not None:
        app.dependency_overrides[get_exam_source_collection] = lambda: sources
    return TestClient(app), calls


def test_word_explanation_verified_past_exam_provenance_and_privacy(caplog) -> None:
    """POST /api/words/explain returns detailed meanings, nuances, and server-side past-exam provenance without logging."""
    caplog.set_level(logging.DEBUG)
    client, calls = client_for(DETAILED_CONTENT)

    secret_context = "秘密の会議の前に必要な資料を準備しておいてください。"
    response = client.post(
        "/api/words/explain",
        json={
            "word": "準備",
            "context": secret_context,
            "explanation_language": "ko",
            "level": "N4",
        },
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"

    body = response.json()
    assert body["word"] == "準備"
    assert body["reading"] == "じゅんび"
    assert body["level"] == "N4"
    assert body["explanation_language"] == "Korean"
    assert body["context_fit_explanation"] == DETAILED_CONTENT["context_fit_explanation"]
    assert "approximate" in body["approximate_difficulty_label"].lower()
    assert "original ai-generated" in body["original_examples_label"].lower()

    # Check meaning, collocations, grammar patterns, and original AI example labeling
    assert len(body["meanings"]) == 1
    first_meaning = body["meanings"][0]
    assert first_meaning["collocations"] == ["準備をする", "心の準備", "準備が整う"]
    assert first_meaning["grammar_patterns"] == ["〜の準備をする", "〜を準備しておく"]
    assert first_meaning["examples"][0]["ai_generated"] is True
    assert "original ai-generated" in first_meaning["examples"][0]["source_label"].lower()

    # Check structured nuances
    assert body["register_and_formality"] == DETAILED_CONTENT["register_and_formality"]
    assert len(body["synonyms_and_contrasts"]) == 2
    assert len(body["natural_vs_unnatural"]) == 1

    # Check server-attached verified past-exam examples from curated JSON collection
    assert body["verified_examples_available"] is True
    assert len(body["verified_past_exam_examples"]) == 1
    ver = body["verified_past_exam_examples"][0]
    assert ver["word"] == "準備"
    assert ver["japanese"] == "会議の前に必要な資料を準備しておいてください。"
    assert ver["source_title"] == "Owner-Curated JLPT Past Exam Archive (Permitted Excerpts)"
    assert "2008 Level 3 Vocabulary, Question 12, p. 6" in ver["source_reference"]
    assert ver["exam_year"] == 2008
    assert ver["exam_level"] == "N4"
    assert ver["question_or_page_ref"] == "Vocabulary Section 2, Question 12, p. 6"
    assert ver["source_type"] == "past_exam"
    assert ver["verified_from_curated_collection"] is True
    assert ver["ai_generated"] is False

    # Verify prompt data boundary and single model invocation
    assert len(calls) == 1
    assert "<untrusted_word_request_data>" in calls[0][1].content
    assert secret_context in calls[0][1].content

    # Verify zero content logging
    assert secret_context not in caplog.text
    assert "準備" not in caplog.text
    assert "/api/words/explain" in client.get("/openapi.json").json()["paths"]


def test_official_practice_samples_are_never_labeled_as_past_exam_questions() -> None:
    """Official practice samples in the curated collection must be excluded from verified_past_exam_examples."""
    client, _ = client_for(MINIMAL_CONTENT)
    # "天気" in app/data/verified_exam_sources.json has source_type="official_practice_sample"
    response = client.post(
        "/api/words/explain",
        json={"word": "天気", "level": "N5", "explanation_language": "English"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified_examples_available"] is False
    assert body["verified_past_exam_examples"] == []
    assert "official practice sample" in body["verified_examples_status"].lower()
    assert "not labeled as past-exam questions" in body["verified_examples_status"].lower()


def test_empty_source_collection_and_unmatched_word_behavior(tmp_path: Path) -> None:
    """Service works when the curated source collection is empty or has no matching entry."""
    # Verify loader handles missing or empty JSON file gracefully
    assert load_curated_exam_sources(tmp_path / "nonexistent.json") == []
    empty_json = tmp_path / "empty.json"
    empty_json.write_text('{"entries": []}', encoding="utf-8")
    assert load_curated_exam_sources(empty_json) == []

    # 1. Empty collection injected
    client_empty, _ = client_for(MINIMAL_CONTENT, sources=[])
    res_empty = client_empty.post("/api/words/explain", json={"word": "準備"})
    assert res_empty.status_code == 200
    body_empty = res_empty.json()
    assert body_empty["verified_examples_available"] is False
    assert body_empty["verified_past_exam_examples"] == []
    assert "empty" in body_empty["verified_examples_status"].lower()

    # 2. Non-empty collection, but word has no verified example
    client_unmatched, _ = client_for(MINIMAL_CONTENT)
    res_unmatched = client_unmatched.post("/api/words/explain", json={"word": "優しい"})
    assert res_unmatched.status_code == 200
    body_unmatched = res_unmatched.json()
    assert body_unmatched["verified_examples_available"] is False
    assert body_unmatched["verified_past_exam_examples"] == []
    assert "no verified past jlpt exam examples" in body_unmatched["verified_examples_status"].lower()


def test_generated_text_cannot_become_verified_exam_citation() -> None:
    """Model output attempting to inject verified_past_exam_examples or fake citations is rejected."""
    injected_fake_citation = {
        **MINIMAL_CONTENT,
        "verified_past_exam_examples": [
            {
                "source_id": "hallucinated-2024-n1",
                "word": "優しい",
                "japanese": "AIが勝手に作った過去問の文です。",
                "source_title": "Hallucinated JLPT 2024 N1",
                "source_reference": "Fake Question 99",
                "exam_year": 2024,
                "exam_level": "N1",
            }
        ],
    }
    client, calls = client_for(injected_fake_citation)
    response = client.post("/api/words/explain", json={"word": "優しい"})
    assert response.status_code == 502
    assert response.headers["cache-control"] == "no-store"
    assert len(calls) == 1


def test_unknown_word_and_ambiguous_reading_acknowledgment() -> None:
    """Supports acknowledging unknown words and ambiguous readings."""
    unknown_word_output = {
        "reading": "ふめいご",
        "is_known_word": False,
        "alternate_readings": ["みかくにんご"],
        "uncertainty_note": "This expression is not found in standard Japanese dictionaries; reading and meaning are uncertain.",
        "meanings": [
            {
                "meaning": "Unverified or coined compound expression",
                "part_of_speech": "Noun (uncertain)",
                "examples": [
                    {
                        "japanese": "この言葉の意味はまだ確認されていません。",
                        "reading": "このことばのいみはまだかくにんされていません。",
                        "translation": "The meaning of this word has not yet been confirmed.",
                        "usage_note": "Used when discussing an unverified term.",
                    }
                ],
            }
        ],
        "nuances": [
            "Because the word is unrecognized, verify the kanji or surrounding context."
        ],
        "follow_up_questions": [
            "この言葉の正しい読み方は何ですか。",
            "似た意味の確認済みの言葉はありますか。",
            "どの辞書でこの言葉を確認できますか。",
        ],
    }
    client, _ = client_for(unknown_word_output)
    res = client.post(
        "/api/words/explain",
        json={"word": "未確認造語X", "explanation_language": "Japanese"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["is_known_word"] is False
    assert body["alternate_readings"] == ["みかくにんご"]
    assert "uncertain" in body["uncertainty_note"].lower()
    assert body["explanation_language"] == "Japanese"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"word": "   "},
        {"word": "a" * 101},
        {"word": "猫", "context": "a" * 501},
        {"word": "猫", "extra": True},
        {"word": "猫", "level": "N6"},
    ],
)
def test_invalid_requests(payload: dict[str, Any]) -> None:
    client, calls = client_for()
    response = client.post("/api/words/explain", json=payload)
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert not calls


@pytest.mark.parametrize(
    "output",
    [
        {},
        {**MINIMAL_CONTENT, "nuances": []},
        {**MINIMAL_CONTENT, "reading": "   "},
        {
            **MINIMAL_CONTENT,
            "meanings": [
                MINIMAL_CONTENT["meanings"][0],
                MINIMAL_CONTENT["meanings"][0],
            ],
        },
        # Follow-up questions are required: missing, too few/many, duplicate, blank or oversized.
        {k: v for k, v in MINIMAL_CONTENT.items() if k != "follow_up_questions"},
        {**MINIMAL_CONTENT, "follow_up_questions": []},
        {**MINIMAL_CONTENT, "follow_up_questions": ["Why?", "How?"]},
        {**MINIMAL_CONTENT, "follow_up_questions": ["Why?", "How?", "When?", "Where?"]},
        {**MINIMAL_CONTENT, "follow_up_questions": ["Why?", "why?", "How?"]},
        {**MINIMAL_CONTENT, "follow_up_questions": ["Why?", "   ", "How?"]},
        {**MINIMAL_CONTENT, "follow_up_questions": ["Why?", "How?", "x" * 501]},
    ],
)
def test_malformed_outputs(output: dict[str, Any]) -> None:
    client, calls = client_for(output)
    response = client.post("/api/words/explain", json={"word": "猫"})
    assert response.status_code == 502
    assert len(calls) == 1
    assert response.headers["cache-control"] == "no-store"


def test_word_explanation_returns_follow_up_questions_in_one_model_call() -> None:
    """Explanation and follow-up questions come from a single model invocation."""
    client, calls = client_for(DETAILED_CONTENT)
    response = client.post(
        "/api/words/explain",
        json={"word": "準備", "level": "N4", "explanation_language": "Korean"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"

    body = response.json()
    assert body["follow_up_questions"] == DETAILED_CONTENT["follow_up_questions"]
    assert len(body["follow_up_questions"]) == 3
    assert len(set(body["follow_up_questions"])) == 3

    # Exactly one model call produced the explanation and the follow-ups together.
    assert len(calls) == 1
    system_prompt = calls[0][0].content
    assert "follow_up_questions" in system_prompt
    assert "exactly 3 distinct" in system_prompt
    assert "Korean" in system_prompt


def test_word_follow_up_questions_are_not_verified_citations_or_logged(caplog) -> None:
    """Follow-ups are plain generated text: never provenance, and never logged."""
    caplog.set_level(logging.DEBUG)
    secret = "SECRET_WORD_FOLLOW_UP_QUESTION_555?"
    output = {
        **DETAILED_CONTENT,
        "follow_up_questions": [secret, "Second question?", "Third question?"],
    }
    client, _ = client_for(output, sources=[])
    response = client.post("/api/words/explain", json={"word": "準備"})
    assert response.status_code == 200
    body = response.json()
    assert secret in body["follow_up_questions"]
    assert body["verified_past_exam_examples"] == []
    assert body["verified_examples_available"] is False
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "error,status",
    [
        (asyncio.TimeoutError(), 504),
        (httpx.ConnectError("private content"), 503),
        (RuntimeError("429 rate limit exceeded"), 429),
        (RuntimeError("private content"), 502),
    ],
)
def test_upstream_errors(error: BaseException, status: int) -> None:
    client, calls = client_for(error=error)
    response = client.post("/api/words/explain", json={"word": "猫"})
    assert response.status_code == status
    assert "private content" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert len(calls) == 1


def test_word_explanation_panel_in_browser_ui() -> None:
    """Browser UI includes the word explanation panel and distinct sections."""
    client = TestClient(create_app())
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    html = res.text

    assert "4. Word Explanation (Meanings, Examples &amp; Nuances)" in html
    assert 'id="word-input"' in html
    assert 'id="word-context-input"' in html
    assert 'id="word-level"' in html
    assert 'id="word-explanation-language"' in html
    assert 'id="word-explanation-results"' in html
    assert "function explainWord()" in html
    assert "function renderWordExplanation()" in html
    assert "function clearWordExplanation()" in html
    assert "word-meanings-section" in html
    assert "word-original-examples-section" in html
    assert "word-verified-examples-section" in html
    assert "word-nuances-section" in html
    assert "word-follow-up-section" in html
