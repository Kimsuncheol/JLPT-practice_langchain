"""FastAPI routes for health checks, JLPT practice, tutor chat, and in-memory browser UI."""

import os
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import HTMLResponse

from app.config import Settings, enforce_privacy_environment, get_settings
from app.schemas import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    HealthResponse,
    PracticeQuestionRequest,
    PracticeQuestionResponse,
    TutorChatRequest,
    TutorChatResponse,
)
from app.services import (
    MissingCredentialsError,
    evaluate_practice_answer,
    generate_practice_question,
    run_tutor_chat,
)

router = APIRouter()


def apply_no_store_headers(response: Response) -> None:
    """Attach mandatory Cache-Control: no-store header to practice and tutor responses."""
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Lightweight health check that never invokes Gemini or external network calls."""
    enforce_privacy_environment()
    tracing_disabled = (
        os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "false"
        and os.environ.get("LANGSMITH_TRACING", "false").lower() == "false"
    )
    return HealthResponse(
        status="ok",
        model=settings.gemini_model,
        tracing_disabled=tracing_disabled,
        persistence="none",
    )


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
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
        # Never log prompts, answers, or exception details containing user content
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate practice question from upstream model.",
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
    """Evaluate a submitted JLPT practice answer with Cache-Control: no-store."""
    apply_no_store_headers(response)
    try:
        return await evaluate_practice_answer(payload, settings=settings)
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to evaluate practice answer.",
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
    settings: Settings = Depends(get_settings),
) -> TutorChatResponse:
    """Stateless JLPT tutor endpoint. Context stays only in browser & request-local memory."""
    apply_no_store_headers(response)
    try:
        return await run_tutor_chat(payload, settings=settings)
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to process tutor chat request.",
            headers={"Cache-Control": "no-store"},
        ) from None


BROWSER_APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>JLPT Practice & Tutor (Ephemeral Mode)</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1e293b; }
    .card { border: 1px solid #cbd5e1; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; background: #f8fafc; }
    .row { display: flex; gap: 0.75rem; margin-bottom: 0.75rem; flex-wrap: wrap; }
    select, input, button, textarea { padding: 0.5rem; border-radius: 6px; border: 1px solid #94a3b8; font-size: 0.95rem; }
    button { background: #2563eb; color: white; border: none; cursor: pointer; font-weight: 600; }
    button:hover { background: #1d4ed8; }
    #chat-box { border: 1px solid #e2e8f0; background: #fff; padding: 1rem; border-radius: 6px; min-height: 160px; max-height: 320px; overflow-y: auto; margin-bottom: 0.75rem; }
    .badge { font-size: 0.8rem; background: #dbeafe; color: #1e40af; padding: 0.2rem 0.5rem; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>JLPT Practice & Tutor <span class="badge">Ephemeral / No-Store</span></h1>
  <p>Conversation context is kept strictly in JavaScript runtime memory (never in localStorage, sessionStorage, cookies, or backend storage) and cleared on page reload.</p>

  <div class="card">
    <h2>1. JLPT Practice Question</h2>
    <div class="row">
      <select id="level">
        <option value="N5">JLPT N5</option>
        <option value="N4">JLPT N4</option>
        <option value="N3">JLPT N3</option>
        <option value="N2">JLPT N2</option>
        <option value="N1">JLPT N1</option>
      </select>
      <select id="category">
        <option value="grammar">Grammar</option>
        <option value="vocabulary">Vocabulary</option>
        <option value="kanji">Kanji</option>
        <option value="reading">Reading</option>
      </select>
      <input id="topic" type="text" placeholder="Optional topic (e.g., particles)" />
      <button type="button" onclick="generateQuestion()">Generate Question</button>
    </div>
    <pre id="practice-output" style="white-space: pre-wrap;"></pre>
  </div>

  <div class="card">
    <h2>2. Ephemeral Tutor Chat</h2>
    <div id="chat-box"></div>
    <div class="row">
      <input id="chat-input" type="text" placeholder="Ask a JLPT grammar or vocabulary question..." style="flex:1;" />
      <button type="button" onclick="sendChat()">Send</button>
      <button type="button" onclick="clearChat()" style="background:#475569;">Clear Memory</button>
    </div>
  </div>

  <script>
    // Ephemeral browser-memory conversation state only (no Web Storage API used)
    let conversationHistory = [];

    async function generateQuestion() {
      const level = document.getElementById('level').value;
      const category = document.getElementById('category').value;
      const topic = document.getElementById('topic').value || null;
      const out = document.getElementById('practice-output');
      out.textContent = 'Generating...';
      const res = await fetch('/api/practice/question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({ level, category, topic })
      });
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
    }

    function renderChat() {
      const box = document.getElementById('chat-box');
      box.innerHTML = '';
      for (const msg of conversationHistory) {
        const div = document.createElement('div');
        div.style.marginBottom = '0.5rem';
        const roleLabel = msg.role === 'user' ? 'You' : 'Tutor';
        div.textContent = `${roleLabel}: ${msg.content}`;
        box.appendChild(div);
      }
      box.scrollTop = box.scrollHeight;
    }

    async function sendChat() {
      const input = document.getElementById('chat-input');
      const message = input.value.trim();
      if (!message) return;
      const level = document.getElementById('level').value;
      input.value = '';

      const res = await fetch('/api/tutor/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({ level, message, history: conversationHistory })
      });
      const data = await res.json();
      if (res.ok && data.reply) {
        conversationHistory.push({ role: 'user', content: message });
        conversationHistory.push({ role: 'assistant', content: data.reply });
        renderChat();
      }
    }

    function clearChat() {
      conversationHistory = [];
      renderChat();
    }
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(response: Response) -> HTMLResponse:
    """Serve minimal single-page UI that stores conversation history only in JS memory."""
    apply_no_store_headers(response)
    return HTMLResponse(content=BROWSER_APP_HTML, headers={"Cache-Control": "no-store"})
