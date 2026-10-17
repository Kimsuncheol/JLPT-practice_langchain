"""FastAPI routes for health checks, JLPT practice generation, local quiz feedback, tutor chat, and UI."""

import asyncio
import os
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import HTMLResponse

from app.config import Settings, enforce_privacy_environment, get_settings
from app.schemas import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    HealthResponse,
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
    MissingCredentialsError,
    StructuralValidationError,
    answer_practice_item_tutor_question,
    evaluate_practice_answer,
    generate_practice_batch,
    generate_practice_question,
    get_chat_model,
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
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except StructuralValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Practice question generation timed out.",
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
        # Never log prompts, answers, or exception details containing user content
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate practice questions from upstream model.",
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
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
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
    """Evaluate a submitted JLPT multiple-choice answer locally without calling Gemini."""
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
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except StructuralValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from None
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Tutor request timed out.",
            headers={"Cache-Control": "no-store"},
        ) from None
    except Exception:
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
  <title>JLPT Practice & Tutor (Unofficial AI Self-Study)</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1e293b; }
    .top-bar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem; }
    .card { border: 1px solid #cbd5e1; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; background: #f8fafc; }
    .passage-card { border: 1px solid #93c5fd; border-radius: 8px; padding: 1rem; margin-top: 1rem; background: #eff6ff; white-space: pre-wrap; }
    .question-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-top: 1rem; background: #ffffff; }
    .row { display: flex; gap: 0.75rem; margin-bottom: 0.75rem; flex-wrap: wrap; align-items: center; }
    select, input[type="text"], button { padding: 0.5rem 0.75rem; border-radius: 6px; border: 1px solid #94a3b8; font-size: 0.95rem; }
    button { background: #2563eb; color: white; border: none; cursor: pointer; font-weight: 600; }
    button:hover { background: #1d4ed8; }
    button.secondary { background: #475569; }
    button.secondary:hover { background: #334155; }
    button.danger { background: #dc2626; }
    button.danger:hover { background: #b91c1c; }
    .badge { font-size: 0.8rem; background: #dbeafe; color: #1e40af; padding: 0.2rem 0.55rem; border-radius: 4px; font-weight: 600; }
    .notice { font-size: 0.88rem; color: #475569; background: #f1f5f9; border-left: 4px solid #64748b; padding: 0.6rem 0.8rem; margin: 0.75rem 0; border-radius: 4px; }
    .option-label { display: block; padding: 0.45rem 0.6rem; margin: 0.35rem 0; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer; }
    .option-label:hover { background: #f8fafc; }
    .feedback-panel { margin-top: 0.85rem; padding: 0.85rem; border-radius: 6px; background: #f8fafc; border: 1px solid #cbd5e1; }
    .correct-banner { color: #166534; font-weight: 700; }
    .incorrect-banner { color: #991b1b; font-weight: 700; }
    #chat-box { border: 1px solid #e2e8f0; background: #fff; padding: 1rem; border-radius: 6px; min-height: 160px; max-height: 320px; overflow-y: auto; margin-bottom: 0.75rem; }
  </style>
</head>
<body>
  <div class="top-bar">
    <h1>JLPT Practice &amp; Tutor <span class="badge">Unofficial AI Practice / Approximate Difficulty</span></h1>
    <button type="button" id="reset-session-btn" class="danger" onclick="resetSession()">Reset Session</button>
  </div>
  <div class="notice">
    <strong>Unofficial AI Practice &amp; Self-Study Notice:</strong> All questions and reading passages are unofficial AI-generated practice material with approximate JLPT difficulty. Practice accuracy is for self-study feedback only and does not predict an official JLPT score. State is kept exclusively in browser JavaScript runtime memory (never saved to <code>localStorage</code>, <code>sessionStorage</code>, cookies, or any backend database). Multiple-choice grading compares your selections locally in the browser against the inspectable answer key without calling Gemini, and is not tamper-proof.
  </div>

  <div class="card">
    <h2>1. JLPT Practice Quiz (Vocabulary, Grammar &amp; Reading)</h2>
    <div class="row">
      <select id="level" aria-label="JLPT Level">
        <option value="N5">JLPT N5 (Approx.)</option>
        <option value="N4">JLPT N4 (Approx.)</option>
        <option value="N3">JLPT N3 (Approx.)</option>
        <option value="N2">JLPT N2 (Approx.)</option>
        <option value="N1">JLPT N1 (Approx.)</option>
      </select>
      <select id="category" aria-label="Practice Category" onchange="syncQuestionCountLimit()">
        <option value="grammar">Grammar</option>
        <option value="vocabulary">Vocabulary</option>
        <option value="reading">Reading (Passage + up to 3 Qs)</option>
      </select>
      <select id="question_count" aria-label="Question Count">
        <option value="1">1 Question</option>
        <option value="2">2 Questions</option>
        <option value="3">3 Questions</option>
        <option value="4">4 Questions</option>
        <option value="5">5 Questions</option>
      </select>
      <select id="explanation_language" aria-label="Explanation Language">
        <option value="English">English</option>
        <option value="Korean">Korean</option>
        <option value="Japanese">Japanese</option>
      </select>
      <button type="button" onclick="generatePracticeSet()">Generate Practice Set</button>
    </div>

    <div id="quiz-status"></div>
    <div id="quiz-container"></div>
    <div id="quiz-actions" class="row" style="margin-top: 1rem; display: none;">
      <button type="button" id="submit-quiz-btn" onclick="submitQuizLocally()">Submit Answers</button>
    </div>
    <div id="quiz-score-summary"></div>
  </div>

  <div class="card">
    <h2>2. Ask Tutor About Current Practice Item</h2>
    <div class="row">
      <label for="tutor-question-select"><strong>Selected Practice Item:</strong></label>
      <select id="tutor-question-select" onchange="selectTutorQuestion(this.value)">
        <option value="0">Question 1 (Generate a practice set first)</option>
      </select>
    </div>
    <div id="chat-box"></div>
    <div class="row">
      <input id="chat-input" type="text" maxlength="500" placeholder="Ask for clarification, grammar comparison, or extra examples..." style="flex:1;" />
      <button type="button" onclick="sendTutorQuestion()">Ask Tutor</button>
      <button type="button" class="secondary" onclick="resetSession()">Reset Session</button>
    </div>
  </div>

  <script>
    // Ephemeral browser-memory state ONLY (never persisted to Web Storage or backend)
    const MAX_RECENT_MESSAGES = 6;
    let currentPracticeSet = null;
    let selectedAnswers = {};
    let isQuizSubmitted = false;
    let activeTutorQuestionIndex = 0;
    let conversationHistory = [];

    function syncQuestionCountLimit() {
      const cat = document.getElementById('category').value;
      const countSelect = document.getElementById('question_count');
      const maxAllowed = cat === 'reading' ? 3 : 5;
      Array.from(countSelect.options).forEach((opt) => {
        const val = parseInt(opt.value, 10);
        opt.disabled = val > maxAllowed;
      });
      if (parseInt(countSelect.value, 10) > maxAllowed) {
        countSelect.value = String(maxAllowed);
      }
    }

    function updateTutorQuestionSelector() {
      const sel = document.getElementById('tutor-question-select');
      sel.innerHTML = '';
      if (!currentPracticeSet || !currentPracticeSet.questions || currentPracticeSet.questions.length === 0) {
        const opt = document.createElement('option');
        opt.value = '0';
        opt.textContent = 'No active question (generate a practice set first)';
        sel.appendChild(opt);
        return;
      }
      currentPracticeSet.questions.forEach((q, idx) => {
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.textContent = `Q${idx + 1}: ${q.question_text}`;
        opt.selected = idx === activeTutorQuestionIndex;
        sel.appendChild(opt);
      });
    }

    function selectTutorQuestion(indexVal) {
      activeTutorQuestionIndex = parseInt(indexVal, 10) || 0;
      const sel = document.getElementById('tutor-question-select');
      if (sel) sel.value = String(activeTutorQuestionIndex);
    }

    async function generatePracticeSet() {
      syncQuestionCountLimit();
      const level = document.getElementById('level').value;
      const category = document.getElementById('category').value;
      const question_count = parseInt(document.getElementById('question_count').value, 10);
      const explanation_language = document.getElementById('explanation_language').value;
      const statusEl = document.getElementById('quiz-status');

      currentPracticeSet = null;
      selectedAnswers = {};
      isQuizSubmitted = false;
      activeTutorQuestionIndex = 0;
      document.getElementById('quiz-container').innerHTML = '';
      document.getElementById('quiz-score-summary').innerHTML = '';
      document.getElementById('quiz-actions').style.display = 'none';
      statusEl.textContent = 'Generating unofficial AI practice set...';

      try {
        const res = await fetch('/api/practice/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-store',
          body: JSON.stringify({ level, category, question_count, explanation_language })
        });
        const data = await res.json();
        if (!res.ok) {
          statusEl.textContent = data.detail || 'Failed to generate practice set.';
          return;
        }
        currentPracticeSet = data;
        statusEl.textContent = `${data.content_label} (${data.difficulty_label}) — Select an answer for each question, then click Submit Answers.`;
        updateTutorQuestionSelector();
        renderQuiz();
      } catch (err) {
        statusEl.textContent = 'Network error while generating practice set.';
      }
    }

    function selectOption(questionId, optionIndex) {
      if (isQuizSubmitted) return;
      selectedAnswers[questionId] = optionIndex;
    }

    function compareAnswerLocally(question, selectedIndex) {
      const isCorrect = selectedIndex === question.correct_index;
      const correctOptionText = question.correct_answer;
      const whyCorrectFits = question.option_explanations[question.correct_index];
      const whyAlternativesDoNotFit = [];

      question.options.forEach((optText, idx) => {
        if (idx !== question.correct_index) {
          whyAlternativesDoNotFit.push({
            optionIndex: idx,
            optionText: optText,
            wasSelected: idx === selectedIndex,
            explanation: question.option_explanations[idx]
          });
        }
      });

      return {
        isCorrect,
        selectedIndex,
        selectedOptionText: typeof selectedIndex === 'number' ? question.options[selectedIndex] : null,
        correctIndex: question.correct_index,
        correctOptionText,
        supportingSentence: question.supporting_sentence || null,
        whyCorrectFits,
        whyAlternativesDoNotFit
      };
    }

    function submitQuizLocally() {
      if (!currentPracticeSet || !currentPracticeSet.questions) return;
      const unanswered = currentPracticeSet.questions.filter(
        (q) => typeof selectedAnswers[q.id] !== 'number'
      );
      if (unanswered.length > 0) {
        document.getElementById('quiz-status').textContent =
          `Please select an option for all ${currentPracticeSet.questions.length} question(s) before submitting.`;
        return;
      }

      // Local grading in browser memory without calling Gemini or any backend endpoint
      isQuizSubmitted = true;
      renderQuiz();
    }

    function renderQuiz() {
      const container = document.getElementById('quiz-container');
      const actionsEl = document.getElementById('quiz-actions');
      const summaryEl = document.getElementById('quiz-score-summary');
      container.innerHTML = '';
      summaryEl.innerHTML = '';

      if (!currentPracticeSet || !currentPracticeSet.questions) {
        actionsEl.style.display = 'none';
        return;
      }

      if (currentPracticeSet.passage) {
        const passageBox = document.createElement('div');
        passageBox.className = 'passage-card';
        const pTitle = document.createElement('div');
        pTitle.innerHTML = '<strong>Reading Passage</strong> <span class="badge">Unofficial AI Reading (Approx. Difficulty)</span>';
        passageBox.appendChild(pTitle);
        const pBody = document.createElement('p');
        pBody.textContent = currentPracticeSet.passage;
        passageBox.appendChild(pBody);
        container.appendChild(passageBox);
      }

      let correctCount = 0;

      currentPracticeSet.questions.forEach((q, qIdx) => {
        const card = document.createElement('div');
        card.className = 'question-card';

        const header = document.createElement('div');
        header.innerHTML = `<strong>Q${qIdx + 1}.</strong> `;
        const qText = document.createElement('span');
        qText.textContent = q.question_text;
        header.appendChild(qText);

        const aiTag = document.createElement('span');
        aiTag.className = 'badge';
        aiTag.style.marginLeft = '0.5rem';
        aiTag.textContent = 'Unofficial AI (Approx. Level)';
        header.appendChild(aiTag);

        const askBtn = document.createElement('button');
        askBtn.type = 'button';
        askBtn.className = 'secondary';
        askBtn.style.marginLeft = '0.5rem';
        askBtn.style.fontSize = '0.8rem';
        askBtn.style.padding = '0.2rem 0.5rem';
        askBtn.textContent = 'Focus in Tutor';
        askBtn.onclick = () => selectTutorQuestion(qIdx);
        header.appendChild(askBtn);

        card.appendChild(header);

        q.options.forEach((optText, optIdx) => {
          const label = document.createElement('label');
          label.className = 'option-label';
          const radio = document.createElement('input');
          radio.type = 'radio';
          radio.name = `question-${q.id}`;
          radio.value = String(optIdx);
          radio.checked = selectedAnswers[q.id] === optIdx;
          radio.disabled = isQuizSubmitted;
          radio.onchange = () => selectOption(q.id, optIdx);

          label.appendChild(radio);
          label.appendChild(document.createTextNode(` ${optIdx + 1}. ${optText}`));
          card.appendChild(label);
        });

        // Reveal explanations ONLY after submission
        if (isQuizSubmitted) {
          const evalResult = compareAnswerLocally(q, selectedAnswers[q.id]);
          if (evalResult.isCorrect) correctCount += 1;

          const feedbackPanel = document.createElement('div');
          feedbackPanel.className = 'feedback-panel';

          const statusBanner = document.createElement('div');
          statusBanner.className = evalResult.isCorrect ? 'correct-banner' : 'incorrect-banner';
          statusBanner.textContent = evalResult.isCorrect
            ? '✓ Correct'
            : `✗ Incorrect (You selected: ${evalResult.selectedOptionText})`;
          feedbackPanel.appendChild(statusBanner);

          const correctLine = document.createElement('p');
          correctLine.innerHTML = '<strong>Correct Answer:</strong> ';
          correctLine.appendChild(
            document.createTextNode(`${evalResult.correctIndex + 1}. ${evalResult.correctOptionText}`)
          );
          feedbackPanel.appendChild(correctLine);

          if (evalResult.supportingSentence) {
            const supportLine = document.createElement('p');
            supportLine.innerHTML = '<strong>Supporting Sentence from Passage:</strong> ';
            supportLine.appendChild(document.createTextNode(evalResult.supportingSentence));
            feedbackPanel.appendChild(supportLine);
          }

          const whyFitsLine = document.createElement('p');
          whyFitsLine.innerHTML = '<strong>Why it fits:</strong> ';
          whyFitsLine.appendChild(document.createTextNode(evalResult.whyCorrectFits));
          feedbackPanel.appendChild(whyFitsLine);

          const altHeader = document.createElement('p');
          altHeader.innerHTML = '<strong>Why the alternatives do not fit:</strong>';
          feedbackPanel.appendChild(altHeader);

          const altList = document.createElement('ul');
          evalResult.whyAlternativesDoNotFit.forEach((alt) => {
            const li = document.createElement('li');
            const prefix = alt.wasSelected ? '[Your choice] ' : '';
            li.textContent = `${prefix}${alt.optionIndex + 1}. ${alt.optionText} — ${alt.explanation}`;
            altList.appendChild(li);
          });
          feedbackPanel.appendChild(altList);

          card.appendChild(feedbackPanel);
        }

        container.appendChild(card);
      });

      actionsEl.style.display = isQuizSubmitted ? 'none' : 'flex';

      if (isQuizSubmitted) {
        const total = currentPracticeSet.questions.length;
        const scoreBox = document.createElement('div');
        scoreBox.className = 'notice';
        scoreBox.textContent =
          `Self-Study Score: ${correctCount} / ${total} — Unofficial AI practice (approximate difficulty). Graded locally in browser memory against the inspectable answer key (not tamper-proof, not stored, and does not predict an official JLPT score).`;
        summaryEl.appendChild(scoreBox);
      }
    }

    function getSelectedQuestionPayload() {
      if (
        currentPracticeSet &&
        currentPracticeSet.questions &&
        currentPracticeSet.questions[activeTutorQuestionIndex]
      ) {
        const q = currentPracticeSet.questions[activeTutorQuestionIndex];
        const selIdx = selectedAnswers[q.id];
        return {
          id: q.id,
          passage: currentPracticeSet.passage || null,
          question_text: q.question_text,
          options: q.options,
          correct_index: q.correct_index,
          correct_answer: q.correct_answer,
          selected_answer: typeof selIdx === 'number' ? q.options[selIdx] : null,
          supporting_sentence: q.supporting_sentence || null,
          option_explanations: q.option_explanations
        };
      }
      return {
        question_text: 'General JLPT practice item inquiry'
      };
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

    async function sendTutorQuestion() {
      const input = document.getElementById('chat-input');
      const message = input.value.trim();
      if (!message) return;
      const level = document.getElementById('level').value;
      const explanation_language = document.getElementById('explanation_language').value;
      const selectedQuestion = getSelectedQuestionPayload();
      const recentWindow = conversationHistory.slice(-MAX_RECENT_MESSAGES);
      input.value = '';

      const res = await fetch('/api/tutor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        body: JSON.stringify({
          level,
          explanation_language,
          question: selectedQuestion,
          message,
          history: recentWindow
        })
      });
      const data = await res.json();
      if (res.ok && data.reply) {
        conversationHistory.push({ role: 'user', content: message });
        conversationHistory.push({ role: 'assistant', content: data.reply });
        if (conversationHistory.length > MAX_RECENT_MESSAGES) {
          conversationHistory = conversationHistory.slice(-MAX_RECENT_MESSAGES);
        }
        renderChat();
      }
    }

    function resetSession() {
      currentPracticeSet = null;
      selectedAnswers = {};
      isQuizSubmitted = false;
      activeTutorQuestionIndex = 0;
      conversationHistory = [];
      document.getElementById('quiz-status').textContent = '';
      document.getElementById('quiz-container').innerHTML = '';
      document.getElementById('quiz-score-summary').innerHTML = '';
      document.getElementById('quiz-actions').style.display = 'none';
      document.getElementById('chat-input').value = '';
      updateTutorQuestionSelector();
      renderChat();
    }
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(response: Response) -> HTMLResponse:
    """Serve minimal single-page UI that stores practice & chat state only in JS memory."""
    apply_no_store_headers(response)
    return HTMLResponse(content=BROWSER_APP_HTML, headers={"Cache-Control": "no-store"})
