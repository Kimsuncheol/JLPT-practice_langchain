# JLPT Practice & Tutor (FastAPI + LangChain + Ollama)

A privacy-first, stateless Japanese Language Proficiency Test (JLPT N5–N1) practice and tutoring web application built with **FastAPI**, **Pydantic**, **Jinja2**, and **`langchain-ollama` (`ChatOllama`)**.

## Project Structure

```text
.
├── .env.example          # Example environment configuration (OLLAMA_BASE_URL & OLLAMA_MODEL)
├── .gitignore            # Git ignore rules (excludes .venv, .env, caches, etc.)
├── README.md             # Local Ollama setup, Render Free deployment, and smoke test instructions
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

## Local Setup Instructions (macOS / Linux)

### 1. Install Ollama and Download the Selected Model

1. **Install Ollama**:
   - macOS (Homebrew):
     ```bash
     brew install --cask ollama
     ```
     *(Or download the installer from [https://ollama.com/download](https://ollama.com/download)).*
2. **Start the local Ollama server**:
   ```bash
   ollama serve
   ```
   By default, Ollama listens at `http://localhost:11434`.
3. **Download (pull) the selected model** (`qwen2.5:7b` is the default and supports Japanese and structured JSON output):
   ```bash
   ollama pull qwen2.5:7b
   ```
   Verify the model is available locally:
   ```bash
   ollama list
   ```

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

Default `.env` values for local development:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_API_KEY=
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

## Render Free Deployment Guide (With Separately Hosted Remote Ollama)

> [!IMPORTANT]
> **Render Free Constraints for Ollama**:
> 1. **Render Free cannot host an Ollama LLM itself** (a Render Free web service has only 512 MB RAM, 0.1 CPU, no GPU, and no persistent disk).
> 2. **Render Free runs in the cloud and cannot access your Mac's `http://localhost:11434`**.
> 3. Therefore, deploying the FastAPI web app on Render Free **requires a separately hosted, network-reachable HTTPS Ollama endpoint** (for example, an Ollama instance running on a cloud GPU VM or private server behind an HTTPS reverse proxy such as Caddy/Nginx/Cloudflare Access with Bearer token authentication).
> 4. If `RENDER=true` and `OLLAMA_BASE_URL` points to `localhost` or `127.0.0.1`, the backend rejects model calls with `503 Service Unavailable` and a clear configuration message.

### 1. Secure Your Remote Ollama Endpoint

- Never expose an unauthenticated Ollama daemon (`0.0.0.0:11434`) directly to the public internet.
- Place your remote Ollama server behind **HTTPS (TLS)** and require an `Authorization: Bearer <token>` header at the reverse proxy.
- Supply that token to the FastAPI backend via the server-side `OLLAMA_API_KEY` environment variable. The FastAPI backend attaches `Authorization: Bearer <OLLAMA_API_KEY>` in `ChatOllama(client_kwargs={"headers": ...})` and never exposes the URL or credentials to the browser.

### 2. Connect Git Repository & Create the Render Service

1. Push this repository to GitHub or GitLab (ensuring `.env` and `.venv/` are ignored).
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** → **Blueprint** and select your repository.
3. Render reads [render.yaml](file:///Users/gimseoncheol/Desktop/JLPT-practice_langchain/render.yaml) to create a single free Python web service (`jlpt-practice-langchain`) with:
   - **Build Command**: `pip install --no-cache-dir -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/health`
   - No database, persistent disk, or background worker.

### 3. Configure Render Environment Variables

| Key | Value / Source | Notes |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | *(Manual Secret / URL)* | Reachable HTTPS URL of your separately hosted Ollama endpoint (e.g., `https://ollama.example.com`). Must **not** be `localhost`. |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Name of the model pulled on the remote Ollama server. |
| `OLLAMA_API_KEY` | *(Manual Secret)* | Server-side Bearer token for authenticating to your secured remote Ollama proxy (`sync: false`). |
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


## Atomic tutor answers and follow-up questions

`POST /api/tutor` and `POST /api/tutor/chat` (including their aliases) now
return `reply` and `follow_up_questions` together. The server asks the model
for both in **one invocation**, validates the result, and returns a single
JSON response. There is no separate suggestion-generation or repair call.

```json
{
  "level": "N5",
  "explanation_language": "English",
  "reply": "The particle は marks the topic of a sentence.",
  "follow_up_questions": [
    "How is が different?",
    "Can you give another example?",
    "How do I make this sentence negative?"
  ]
}
```

The selected-item route also returns its existing `question_id` and
`ai_generated` fields. Follow-ups are exactly three distinct, nonempty strings
of at most 500 characters each, written in `explanation_language` and phrased
as questions the learner can ask. The explanation is nonempty and at most
20,000 characters; it can contain Markdown. An incomplete or malformed model
result returns HTTP 502 with `Cache-Control: no-store`, without retrying the
model. Existing request limits and ephemeral history handling are unchanged.

Redeploy this backend before using the combined response in the Flutter app.
The generated OpenAPI specification and `output/pdf/jlpt-practice-api-reference.pdf`
describe the updated contract.

## Word explanation service

`POST /api/words/explain` (alias `POST /words/explain`) explains a Japanese word or expression using Ollama (`ChatOllama`) and attaches verified past-exam examples strictly from the project owner's local curated JSON source collection (`app/data/verified_exam_sources.json`).
There is no application authentication on this endpoint, matching existing routes.
It makes one bounded model invocation without automatic repair calls. No submitted words, context, or generated explanations are saved or logged by this service.
The browser interface at `GET /` includes **Section 4: Word Explanation (Meanings, Examples & Nuances)**, which renders meanings, original AI-generated examples, verified past-exam examples, and nuances in four clearly distinguished panels while keeping state exclusively in JavaScript runtime memory.

```bash
curl -X POST http://127.0.0.1:8000/api/words/explain \
  -H 'Content-Type: application/json' \
  -d '{"word":"準備","context":"会議の前に必要な資料を準備しておいてください。","level":"N4","explanation_language":"English"}'
```

### Request schema (`WordExplainRequest`)

Unknown fields are rejected (`extra="forbid"`) and surrounding whitespace is trimmed:

| Field | Required / default | Constraint |
| --- | --- | --- |
| `word` | Required | 1–100 characters after trimming |
| `context` | Optional, `null` | Up to 500 characters |
| `level` | `N5` | `N5`, `N4`, `N3`, `N2`, `N1` (approximate difficulty) |
| `explanation_language` | `English` | `English`, `Korean`, `Japanese`; `en`/`ko`/`ja` and case variants accepted |

### Successful response (`WordExplainResponse`, HTTP 200)

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

### Contract & Provenance Guarantees

1. **Meanings & Context Disambiguation**:
   - Returns primary kana `reading`, `alternate_readings` (when ambiguous), `is_known_word`, and `uncertainty_note` (acknowledging unknown words or uncertain readings rather than inventing facts).
   - `meanings` contains 1–4 distinct senses, each with `part_of_speech`, optional `context_fit`, `collocations` (up to 6), `grammar_patterns` (up to 6), and 1–3 original AI-generated `examples`.
   - `context_fit_explanation` explains which meaning fits the optional user-provided `context`.
2. **Original vs. Verified Past-Exam Examples**:
   - Original examples inside `meanings[].examples` are always labeled with `ai_generated: true` and `source_label: "Original AI-generated example (unofficial)"`.
   - `verified_past_exam_examples` is populated **exclusively server-side** from `app/data/verified_exam_sources.json` (`CuratedExamSourceRecord`). The LLM output schema (`WordExplanationContent`) uses `extra="forbid"` and has no field for verified exam examples, so model-generated text can never become a verified exam citation.
   - Each verified past-exam record includes `source_title`, `source_reference`, optional `source_url`, and `exam_year`, `exam_level`, and `question_or_page_ref` when known.
   - Official practice samples (`source_type="official_practice_sample"`) are never labeled or returned as past-exam questions (`source_type="past_exam"`).
   - When the curated collection is empty or has no verified past-exam match for a word, `verified_past_exam_examples` is `[]`, `verified_examples_available` is `false`, and `verified_examples_status` provides a clear availability message.
3. **Nuances & Approximate Difficulty**:
   - Includes `nuances` (1–5 summary points), `register_and_formality`, `synonyms_and_contrasts`, `natural_vs_unnatural` usage pairs, and `approximate_difficulty_label`.
4. **Headers & Error Responses**:
   - All responses (including errors) include `Cache-Control: no-store` and `Pragma: no-cache`.
   - Errors: `413` oversized body, `422` invalid request fields, `429` upstream rate limit, `502` malformed model output or generation failure, `503` unavailable or misconfigured Ollama, and `504` model timeout.

