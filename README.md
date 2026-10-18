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
│   ├── main.py           # FastAPI application entrypoint, StaticFiles, & no-store middleware
│   ├── prompts.py        # Modular LangChain ChatPromptTemplate definitions
│   ├── routes.py         # HTTP endpoints (/health, /api/practice/generate, /api/tutor, Jinja2 UI)
│   ├── schemas.py        # Pydantic v2 request/response schemas & structural validation
│   ├── services.py       # Async ChatOllama services, structured output fallback, & local grading
│   ├── static/
│   │   ├── app.js        # Minimal JS keeping study state exclusively in browser memory
│   │   └── styles.css    # Responsive CSS with Japanese typography & keyboard focus styles
│   └── templates/
│       └── index.html    # Responsive Jinja2 UI template
└── tests/
    └── test_app.py       # Automated tests using a fake model for validation, grading, errors, and privacy
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

