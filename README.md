# JLPT Practice & Tutor (FastAPI + LangChain + Ollama)

A privacy-first, stateless Japanese Language Proficiency Test (JLPT N5–N1) practice and tutoring web application built with **FastAPI**, **Pydantic**, **Jinja2**, and **`langchain-ollama` (`ChatOllama`)**.

## Project Structure

```text
.
├── .env.example          # Example environment configuration (OLLAMA_BASE_URL & OLLAMA_MODEL)
├── .gitignore            # Git ignore rules (excludes .venv, .env, caches, etc.)
├── README.md             # Ollama Cloud setup, Render Free deployment, and smoke test instructions
├── pyproject.toml        # Project metadata and pytest configuration
├── render.yaml           # Render Blueprint for one free Python web service
├── requirements.txt      # Reproducible pinned Python dependencies (no Gemini/Google packages)
├── app/
│   ├── __init__.py
│   ├── config.py         # Ollama environment config & mandatory tracing disablement
│   ├── data/
│   │   └── verified_exam_sources.json # Curated reference collection for verified past-exam examples
│   ├── main.py           # FastAPI application entrypoint, StaticFiles, & no-store middleware
│   ├── prompts.py        # Modular LangChain ChatPromptTemplate definitions
│   ├── routes.py         # HTTP endpoints (/health, /api/practice/*, /api/tutor/*, Jinja2 UI)
│   ├── schemas.py        # Pydantic v2 request/response schemas & structural validation
│   ├── services.py       # Async ChatOllama services, structured output fallback, & local grading
│   ├── words.py          # Word explanation endpoint (/api/words/explain) & curated source provenance
│   ├── static/
│   │   ├── app.js        # Minimal JS keeping study state exclusively in browser memory
│   │   └── styles.css    # Responsive CSS with Japanese typography & keyboard focus styles
│   └── templates/
│       └── index.html    # Responsive Jinja2 UI template (quiz, tutor, & word explanation panel)
└── tests/
    ├── test_app.py       # Automated tests for practice, follow-ups, tutor, grading, and privacy
    └── test_words.py     # Automated tests for word explanations, curated provenance, and UI panel
```

## Mandatory Privacy & Self-Study Guarantees

- **Zero Persistence**: No database, persistent disk, background worker, conversation files, or persistent server-side chat memory.
- **Zero Content/Credential Logging**: Prompts, generated quizzes, learner answers, conversation messages, and credentials are never logged.
- **Tracing Disabled**: `LANGCHAIN_TRACING_V2` and `LANGSMITH_TRACING` are forced to `"false"` and LangSmith variables are stripped at startup and runtime.
- **Ephemeral Memory Only**: Practice sets, selected answers, session accuracy, missed topics, and bounded 6-message tutor context live exclusively in browser JavaScript runtime memory (never `localStorage`, `sessionStorage`, `IndexedDB`, or cookies) and request-local Python variables during a single HTTP call. Refreshing the page or clicking **Reset Session** clears all study state.
- **No-Store Caching**: Every `/practice/*`, `/api/practice/*`, `/tutor/*`, `/api/tutor/*`, and `/` response includes `Cache-Control: no-store`.
- **No Google API Key or Gemini Dependencies**: Uses `langchain-ollama` and `ChatOllama` exclusively.
- **Zero Startup/Health/Grading Model Calls**: Starting the server, calling `GET /health`, or grading multiple-choice questions never invokes Ollama.

---

## Development Setup (macOS / Linux, Ollama Cloud)

### 1. Create an Ollama Cloud API Key

Create a server-side API key at [Ollama API keys](https://ollama.com/settings/keys).
This application calls [Ollama Cloud directly](https://docs.ollama.com/cloud) at
`https://ollama.com`; no local Ollama installation, daemon, or model download is needed.
The default model is `gpt-oss:120b`. Use a model name returned by
`https://ollama.com/api/tags`, rather than a local CLI `-cloud` alias.

### 2. Create Local Python Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set these values in `.env` for the locally running FastAPI server:

```env
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_API_KEY=your_ollama_cloud_api_key
LANGCHAIN_TRACING_V2=false
LANGCHAIN_TRACING=false
LANGSMITH_TRACING=false
```

### 4. Run the Application Locally

```bash
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Browser UI**: `http://127.0.0.1:8000/`
- **Health Check**: `http://127.0.0.1:8000/health`

### 5. Run Automated Tests (Uses Fake Model — No Running Ollama Required)

```bash
.venv/bin/pytest
```

---

## Render Free Deployment Guide (Ollama Cloud)

Render hosts the FastAPI application; inference runs directly on Ollama Cloud.
`OLLAMA_BASE_URL` defaults to `https://ollama.com` and `OLLAMA_MODEL` defaults to
`gpt-oss:120b` in both the application and Render Blueprint.

### 1. Configure Cloud Authentication

- Create an [Ollama Cloud API key](https://ollama.com/settings/keys).
- Set `OLLAMA_API_KEY` in the backend environment. It is required for direct Cloud calls.
- The backend passes `Authorization: Bearer <OLLAMA_API_KEY>` through
  `ChatOllama(client_kwargs={"headers": ...})` to both sync and async Ollama clients.
- Keep the key exclusively on the backend. The example environment file contains no credential.

### 2. Connect Git Repository & Create the Render Service

1. Push this repository to GitHub or GitLab (ensuring `.env` and `.venv/` are ignored).
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** → **Blueprint** and select your repository.
3. Render reads [render.yaml](render.yaml) to create a single free Python web service (`jlpt-practice-langchain`) with:
   - **Build Command**: `pip install --no-cache-dir -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/health`
   - No database, persistent disk, or background worker.

### 3. Configure Render Environment Variables

| Key | Value / Source | Notes |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `https://ollama.com` | Direct Ollama Cloud host; do not append `/api` or `/v1`. |
| `OLLAMA_MODEL` | `gpt-oss:120b` | Cloud API model name from `/api/tags`. |
| `OLLAMA_API_KEY` | *(Manual Secret)* | Ollama Cloud API key, supplied as a server-side Bearer token (`sync: false`). |
| `LANGCHAIN_TRACING_V2` | `false` | Keeps LangChain tracing disabled. |
| `LANGCHAIN_TRACING` | `false` | Keeps LangChain tracing disabled. |
| `LANGSMITH_TRACING` | `false` | Keeps LangSmith tracing disabled. |

---

## Short Manual Smoke Test

Set `BASE_URL="http://127.0.0.1:8000"` locally (or your Render URL when testing a deployed instance):

```bash
# 1. Lightweight health check (does not call Ollama)
curl -i "$BASE_URL/health"

# 2. Generate 2 JLPT N5 grammar questions (Cache-Control: no-store)
curl -i -X POST "$BASE_URL/api/practice/generate" \
  -H "Content-Type: application/json" \
  -d '{"level":"N5","category":"grammar","question_count":2,"explanation_language":"English"}'

# 3. Generate a follow-up question targeting the same learning point ("same" or "harder")
curl -i -X POST "$BASE_URL/api/practice/follow-up" \
  -H "Content-Type: application/json" \
  -d '{
    "level": "N5",
    "category": "grammar",
    "explanation_language": "English",
    "difficulty": "harder",
    "selected_answer": "を",
    "current_item": {
      "question_text": "わたし（　）がくせいです。",
      "options": ["は", "を", "に", "で"],
      "correct_index": 0
    }
  }'

# 4. Ask the tutor about a selected practice item
curl -i -X POST "$BASE_URL/api/tutor" \
  -H "Content-Type: application/json" \
  -d '{
    "level": "N5",
    "explanation_language": "English",
    "question": {
      "question_text": "わたし（　）がくせいです。",
      "options": ["は", "を", "に", "で"],
      "correct_index": 0
    },
    "history": [],
    "message": "Why does は fit better than が here?"
  }'
```


## API Reference & Contract Documentation

This section provides the complete reference for all HTTP endpoints, schemas, privacy guarantees, and error behaviors across the application. Live interactive OpenAPI documentation is also served at `GET /docs` (Swagger UI), `GET /redoc` (ReDoc), and `GET /openapi.json`.

### Endpoints Directory

| Method & Path | Alias | Model Call | Description |
| --- | --- | --- | --- |
| `GET /health` | — | No | Lightweight liveness and configuration check (never calls Ollama). |
| `POST /api/practice/generate` | `/practice/generate` | Yes | Generate 1–5 JLPT practice questions (1–3 for reading with passage). |
| `POST /api/practice/follow-up` | `/practice/follow-up` | Yes | Generate a follow-up question for the same learning point (`same` or `harder`). |
| `POST /api/practice/evaluate` | `/practice/evaluate` | No | Local multiple-choice answer comparison against answer key without calling Ollama. |
| `POST /api/practice/question` | `/practice/question` | Yes | Generate a single structured question with optional topic focus. |
| `POST /api/tutor` | `/tutor` | Yes | Tutor inquiry about a selected question; returns answer & 3 follow-ups in 1 call. |
| `POST /api/tutor/chat` | `/tutor/chat` | Yes | General or item-focused stateless tutor turn using request-local history. |
| `POST /api/words/explain` | `/words/explain` | Yes | Detailed word meanings, original examples, curated past-exam citations, nuances, & 3 follow-up questions (one model call). |
| `GET /` | — | No | Responsive browser interface with in-memory state. |

---

### Global Conventions & Privacy Contract

- **Headers**: All practice, tutor, word, and root responses return `Cache-Control: no-store` and `Pragma: no-cache`.
- **Zero Logging & Persistence**: Prompts, generated questions, learner answers, conversation messages, and credentials are never written to databases, files, logs, caches, or tracing systems.
- **Request Body Bounds**: Enforced by server middleware (`MAX_REQUEST_BODY_BYTES`, default 4,096 bytes; returns HTTP `413` when exceeded).
- **Timeouts**: Handled with `OLLAMA_TIMEOUT_SECONDS` (default 45s; returns HTTP `504` on generation routes).
- **Levels & Languages**:
  - `level`: `N5`, `N4`, `N3`, `N2`, `N1` (approximate difficulty; practice accuracy does not predict official exam scores).
  - `explanation_language`: `English`, `Korean`, `Japanese` (normalizes `en`, `ko`, `ja`, and case variants).

---

### 1. `GET /health`

Lightweight application health and configuration check. Never makes an external network call or invokes Ollama.

**Response (`HealthResponse`, HTTP 200)**:
```json
{
  "status": "ok",
  "model": "gpt-oss:120b",
  "base_url_configured": true,
  "tracing_disabled": true,
  "persistence": "none"
}
```

---

### 2. `POST /api/practice/generate`

Generates 1–5 original JLPT practice questions (1–3 for reading with a short reading passage) in a single async Ollama batch call.

**Request (`PracticeGenerateRequest`)**:
```json
{
  "level": "N5",
  "category": "grammar",
  "question_count": 2,
  "explanation_language": "English"
}
```
*Note: `category` accepts `vocabulary`, `grammar`, or `reading`. `count` is accepted as an alias for `question_count`.*

**Response (`PracticeGenerateResponse`, HTTP 200)**:
```json
{
  "level": "N5",
  "category": "grammar",
  "question_count": 2,
  "explanation_language": "English",
  "passage": null,
  "ai_generated": true,
  "unofficial_practice": true,
  "difficulty_label": "Approximate JLPT difficulty (unofficial AI-generated estimate)",
  "content_label": "Unofficial AI-generated original JLPT practice material (not official exam items)",
  "score_disclaimer": "Unofficial AI practice with approximate difficulty: practice accuracy is for self-study only and does not predict or guarantee an official JLPT score.",
  "self_study_notice": "Self-study mode: The answer key is held in browser memory and compared locally without calling the Ollama model or storing scores. Because a browser-held answer key is inspectable, this scoring is for self-study feedback only, is not tamper-proof, and does not predict an official JLPT score.",
  "validation": {
    "structural_validation_passed": true,
    "linguistic_correctness_guaranteed": false,
    "official_score_predictive": false,
    "notice": "Structural validation passed (schema, Japanese script presence, 4 distinct options, single valid answer index, and per-option explanations verified). Note: Structural validation does not guarantee linguistic nuance or pedagogical accuracy of AI-generated content."
  },
  "questions": [
    {
      "id": "jlpt-n5-grammar-4f71a9b8c2d1",
      "question_text": "わたしは毎朝、パン（　）食べます。",
      "options": ["を", "は", "に", "で"],
      "correct_index": 0,
      "correct_answer": "を",
      "option_explanations": [
        "を marks the direct object of the verb 食べます.",
        "は marks the topic and is unnatural directly following パン in this simple sentence.",
        "に marks destination or specific time.",
        "で marks means or location of an action."
      ],
      "supporting_sentence": null,
      "ai_generated": true,
      "difficulty_label": "Approximate JLPT difficulty (unofficial AI-generated estimate)",
      "content_label": "Unofficial AI-generated original JLPT practice material (not official exam items)"
    }
  ]
}
```

---

### 3. `POST /api/practice/follow-up`

Generates exactly one original question targeting the same core learning point as the previous item, at either `"same"` difficulty ("Practice this again") or `"harder"` difficulty ("Try a harder question"), while strictly avoiding repeating the previous sentence.

**Request (`PracticeFollowUpRequest`)**:
```json
{
  "level": "N4",
  "category": "grammar",
  "difficulty": "harder",
  "explanation_language": "English",
  "selected_answer": "読んで",
  "current_item": {
    "id": "jlpt-n4-grammar-111111111111",
    "question_text": "毎朝、新聞を（　）から会社へ行きます。",
    "options": ["読んで", "読み", "読む", "読んだ"],
    "correct_index": 0,
    "correct_answer": "読んで",
    "option_explanations": ["1", "2", "3", "4"]
  }
}
```
*Aliases accepted for `difficulty`: `same`, `again`, `practice_again`, `similar`, `harder`, `challenge`, `more_difficult`.*

**Response (`PracticeFollowUpResponse`, HTTP 200)**:
Returns the new question with its distinct 4 choices, single correct answer, 4 option explanations, and stable ID derived from content.

---

### 4. `POST /api/practice/evaluate`

Performs deterministic answer grading by comparing the learner's choice with the browser-held answer key. **Does not call Ollama.**

**Request (`AnswerEvaluationRequest`)**:
```json
{
  "level": "N5",
  "question": "わたし（　）がくせいです。",
  "selected_answer": "を",
  "options": ["は", "を", "に", "で"],
  "correct_index": 0,
  "option_explanations": [
    "は marks the sentence topic.",
    "を marks a direct object.",
    "に marks direction or specific time.",
    "で marks means or location of an action."
  ]
}
```

**Response (`AnswerEvaluationResponse`, HTTP 200)**:
```json
{
  "is_correct": false,
  "selected_answer": "を",
  "correct_answer": "は",
  "supporting_sentence": null,
  "why_correct_fits": "は marks the sentence topic.",
  "why_alternatives_do_not_fit": [
    {
      "option_index": 1,
      "option_text": "を",
      "was_selected": true,
      "why_it_does_not_fit": "を marks a direct object."
    },
    {
      "option_index": 2,
      "option_text": "に",
      "was_selected": false,
      "why_it_does_not_fit": "に marks direction or specific time."
    },
    {
      "option_index": 3,
      "option_text": "で",
      "was_selected": false,
      "why_it_does_not_fit": "で marks means or location of an action."
    }
  ],
  "feedback": "Not quite. You selected 'を', but the correct answer is 'は'. は marks the sentence topic.",
  "grammar_notes": [],
  "self_study_mode": true,
  "tamper_proof": false,
  "official_score_predictive": false,
  "difficulty_label": "Approximate JLPT difficulty (unofficial AI-generated estimate)",
  "scoring_notice": "Self-study mode: The answer key is held in browser memory and compared locally without calling the Ollama model or storing scores. Because a browser-held answer key is inspectable, this scoring is for self-study feedback only, is not tamper-proof, and does not predict an official JLPT score.",
  "score_disclaimer": "Unofficial AI practice with approximate difficulty: practice accuracy is for self-study only and does not predict or guarantee an official JLPT score."
}
```

---

### 5. `POST /api/practice/question`

Generates a single question with optional topic focus (e.g. particles, te-form, daily life).

**Request (`PracticeQuestionRequest`)**:
```json
{
  "level": "N4",
  "category": "grammar",
  "topic": "te-form connections"
}
```

**Response (`PracticeQuestionResponse`, HTTP 200)**:
```json
{
  "level": "N4",
  "category": "grammar",
  "question": "朝起きて、顔を（　）、朝ご飯を食べました。",
  "options": ["洗って", "洗い", "洗う", "洗った"],
  "correct_index": 0,
  "explanation": "Te-form is used to connect sequential actions in chronological order."
}
```

---

### 6. `POST /api/tutor` & `POST /api/tutor/chat`

Stateless JLPT tutoring. The server asks Ollama for the concise pedagogical reply and **exactly three distinct follow-up questions** in **one atomic model invocation**.

**Request (`TutorRequest` for `/api/tutor`)**:
```json
{
  "level": "N5",
  "explanation_language": "English",
  "question": {
    "question_text": "わたし（　）がくせいです。",
    "options": ["は", "を", "に", "で"],
    "correct_index": 0
  },
  "message": "Why does は fit better than が here?",
  "history": []
}
```

**Response (`TutorResponse`, HTTP 200)**:
```json
{
  "level": "N5",
  "explanation_language": "English",
  "question_id": null,
  "reply": "The particle は indicates the topic of the sentence ('as for me...'). While が marks a grammatical subject, using は here is natural when introducing yourself or stating your identity.",
  "follow_up_questions": [
    "When should I use が instead of は for identity?",
    "Can you give 2 more examples using は with occupations?",
    "How do I say 'I am not a student'?"
  ],
  "ai_generated": true
}
```
*Note: For `POST /api/tutor/chat` (`TutorChatRequest`), `question` is optional for general tutoring, and `TutorChatResponse` returns `level`, `explanation_language`, `reply`, and `follow_up_questions`.*

---

### 7. `POST /api/words/explain`

Explains a Japanese word or expression with deep grammatical and lexical nuance, original AI examples, server-attached verified past-exam citations from `app/data/verified_exam_sources.json`, and three AI-generated follow-up questions produced in the same single model call.

Generation uses the model's structured-output interface before considering plain
text invocation. Direct Ollama Cloud (`https://ollama.com`) uses function calling
to return validated output arguments: Cloud does not support Ollama's JSON-schema
`format` parameter. Other Ollama hosts use native JSON-schema output. The same
provider selection applies to structured practice generation. A Cloud model must
support tool calling (the default `gpt-oss:120b` does).

Word explanations have a separate bounded output budget, `WORD_NUM_PREDICT`
(default 8,192; range 2,048–16,384), and timeout, `WORD_TIMEOUT_SECONDS`
(default 90 seconds; range 1–120). The generic 2,048-token budget is too small
for reliably completing detailed explanations plus follow-up questions, especially
with reasoning models. `gpt-oss` uses low reasoning effort; the reasoning text is
never part of the API response. There are no automatic retries or repair calls.

**Request (`WordExplainRequest`)**:
```json
{
  "word": "準備",
  "context": "会議の前に必要な資料を準備しておいてください。",
  "level": "N4",
  "explanation_language": "English"
}
```

**Response (`WordExplainResponse`, HTTP 200)**:
```json
{
  "word": "準備",
  "context": "会議の前に必要な資料を準備しておいてください。",
  "level": "N4",
  "explanation_language": "English",
  "reading": "じゅんび",
  "is_known_word": true,
  "alternate_readings": [],
  "uncertainty_note": null,
  "context_fit_explanation": "In the provided meeting context, 準備 refers to preparing materials in advance.",
  "meanings": [
    {
      "meaning": "Preparation; arrangements; getting ready",
      "part_of_speech": "Noun / する verb",
      "context_fit": "Directly fits preparing meeting documents beforehand.",
      "collocations": ["準備をする", "心の準備", "準備が整う"],
      "grammar_patterns": ["〜の準備をする", "〜を準備しておく"],
      "examples": [
        {
          "japanese": "旅行の準備はもう終わりましたか。",
          "reading": "りょこうのじゅんびはもうおわりましたか。",
          "translation": "Have you finished preparing for the trip already?",
          "usage_note": "Used with の to modify 準備 as a noun.",
          "ai_generated": true,
          "source_label": "Original AI-generated example (unofficial)"
        }
      ]
    }
  ],
  "nuances": [
    "準備 focuses on concrete, practical preparations before an event or task.",
    "Compared with 用意 (ようい), 準備 often implies broader multi-step readiness."
  ],
  "register_and_formality": "Neutral register suitable for both everyday conversation and formal business contexts.",
  "synonyms_and_contrasts": [
    {
      "term": "用意 (ようい)",
      "comparison": "用意 often focuses on having specific items ready to use immediately, whereas 準備 covers broader advance preparation."
    }
  ],
  "natural_vs_unnatural": [
    {
      "natural_example": "会議の資料を準備しておいた。",
      "unnatural_example": "会議の資料を身支度しておいた。",
      "explanation": "身支度 is only for personal grooming/dressing, not business documents."
    }
  ],
  "follow_up_questions": [
    "How is 準備 different from 用意?",
    "Can you give another example with 準備をする?",
    "When would I say 準備中 instead?"
  ],
  "approximate_difficulty_label": "Approximate JLPT difficulty (unofficial estimate; no official JLPT vocabulary list is published)",
  "original_examples_label": "Original AI-generated examples (unofficial study material, separate from verified past JLPT exams)",
  "verified_past_exam_examples": [
    {
      "source_id": "jlpt-past-2008-l3-q12",
      "word": "準備",
      "japanese": "会議の前に必要な資料を準備しておいてください。",
      "reading": "かいぎのまえにひつようなしりょうをじゅんびしておいてください。",
      "source_title": "Owner-Curated JLPT Past Exam Archive (Permitted Excerpts)",
      "source_reference": "JEES / Japan Foundation Historical JLPT Past Exam Compilation, 2008 Level 3 Vocabulary, Question 12, p. 6",
      "source_url": "https://www.jlpt.jp/reference/books.html",
      "exam_year": 2008,
      "exam_level": "N4",
      "question_or_page_ref": "Vocabulary Section 2, Question 12, p. 6",
      "source_type": "past_exam",
      "verified_from_curated_collection": true,
      "ai_generated": false
    }
  ],
  "verified_examples_available": true,
  "verified_examples_status": "Found 1 verified past JLPT exam example(s) in the curated source collection.",
  "ai_generated": true,
  "notice": "AI-generated explanation; linguistic accuracy is not guaranteed. JLPT difficulty is approximate rather than an official word-level classification."
}
```

#### Curated Past-Exam Provenance Rules
- `verified_past_exam_examples` is attached **strictly server-side** from `app/data/verified_exam_sources.json`.
- The LLM output schema (`WordExplanationContent`) has `extra="forbid"` and lacks verified exam fields, guaranteeing model-generated text can never become a verified exam citation.
- Records with `source_type="official_practice_sample"` are excluded from `verified_past_exam_examples` so official practice samples are never labeled as past-exam questions.
- If the curated source collection is empty or has no matching record, `verified_past_exam_examples` returns `[]` with a clear availability message in `verified_examples_status`.

#### Follow-Up Question Rules (added field: `follow_up_questions`)
- `follow_up_questions` is a **new required response field** (additive; all previously documented fields are unchanged). Existing clients that ignore unknown fields are unaffected.
- It contains **exactly three distinct, nonblank strings of at most 500 characters**, written in `explanation_language` and phrased as questions the learner could ask next about the word.
- The explanation and the follow-up questions are generated **together in one model invocation**. There is no separate suggestion call and no automatic repair or retry call.
- If the model output is missing the field, or has too few/too many questions, duplicates (compared ignoring case and repeated whitespace), blank entries, or an oversized entry, the whole request fails with HTTP `502` and `Cache-Control: no-store`. A partial explanation is never returned.
- Follow-up questions are plain generated text. They are never provenance and never appear in `verified_past_exam_examples`.
- They are not saved or logged. The browser shows them in section `4e. Suggested Follow-Up Questions (AI-Generated)` and keeps them only in JavaScript memory; they are cleared by **Clear Word** and **Reset Session**.

---

### Error Responses & Client Handling

| HTTP Status | Trigger | Safe Detail / Format |
| --- | --- | --- |
| `413 Content Too Large` | Request body exceeds configured `MAX_REQUEST_BODY_BYTES` (default 4,096 bytes). | `{"detail": "Request body exceeds maximum allowed size of 4096 bytes."}` |
| `422 Unprocessable Entity` | Pydantic validation failure (e.g., question count out of range, missing fields, invalid role). | FastAPI standard `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| `429 Too Many Requests` | Remote Ollama gateway returns rate limit / quota exhaustion. | `{"detail": "Ollama endpoint rate limit exceeded. Please wait and try again."}` |
| `502 Bad Gateway` | Malformed model output, duplicate choices, missing Japanese script, or failed structural validation. | `{"detail": "Failed to generate practice questions from Ollama model."}` |
| `503 Service Unavailable` | Ollama endpoint unreachable, model missing, missing Cloud API key, or invalid `localhost` URL on Render Free. | `{"detail": "Ollama service or selected model is unavailable."}` |
| `504 Gateway Timeout` | Async generation exceeded configured `OLLAMA_TIMEOUT_SECONDS`. | `{"detail": "Practice question generation timed out waiting for Ollama."}` |

For word explanations, `502` means generation failed or the response could not be
validated, with detail `Word explanation failed or returned invalid output.`;
it does not mean the HTTP service is offline. `504` uses `WORD_TIMEOUT_SECONDS`
and detail `Word explanation timed out waiting for Ollama.`. The Flutter client
maps `502` to its localized incomplete-response message and keeps manual retry.

For missing Cloud credentials, generation and tutor routes return HTTP `503` with
`{"detail":"OLLAMA_API_KEY must be configured on the backend for Ollama Cloud."}`.
The word explanation route returns the sanitized HTTP `503` detail
`Ollama service or selected model is unavailable.`. No upstream call is made.
`GET /health` remains a liveness check: it does not validate the key, model access,
or upstream connectivity. A healthy response does not confirm Cloud inference works.

For an existing Render service, update its environment values to the table above
and redeploy; repository changes alone do not verify or update the running service.

All error responses include `Cache-Control: no-store` and `Pragma: no-cache`. Safe, sanitized error messages prevent prompt or credential leakage.


