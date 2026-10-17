# JLPT Practice & Tutor (FastAPI + LangChain + Gemini)

A privacy-first, stateless Japanese Language Proficiency Test (JLPT N5–N1) practice and tutoring web application built with **FastAPI**, **Pydantic**, and **`langchain-google-genai`**.

## Project Structure

```text
.
├── .env.example          # Example environment configuration
├── .gitignore            # Git ignore rules (excludes .venv, .env, caches, etc.)
├── README.md             # Local setup and architecture documentation
├── requirements.txt      # Reproducible pinned Python dependencies
├── app/
│   ├── __init__.py
│   ├── config.py         # Environment config & mandatory tracing disablement
│   ├── main.py           # FastAPI application entrypoint & no-store middleware
│   ├── prompts.py        # Modular LangChain ChatPromptTemplate definitions
│   ├── routes.py         # HTTP endpoints (/health, practice, tutor, browser UI)
│   ├── schemas.py        # Pydantic v2 request/response schemas
│   └── services.py       # Stateless langchain-google-genai service functions
└── tests/
    └── test_app.py       # Automated tests for health, privacy, and endpoints
```

## Mandatory Privacy Guarantees

- **Zero Persistence**: No database, conversation files, or persistent server-side chat memory.
- **Zero Content/Credential Logging**: Prompts, learner answers, conversation messages, and API keys are never logged.
- **Tracing Disabled**: `LANGCHAIN_TRACING_V2` and `LANGSMITH_TRACING` are forced to `"false"` and LangSmith variables are stripped at startup and runtime.
- **Ephemeral Memory Only**: Multi-turn tutor context lives exclusively in browser JavaScript runtime memory (never `localStorage` or `sessionStorage`) and request-local Python variables during a single HTTP call.
- **No-Store Caching**: Every `/practice/*`, `/api/practice/*`, `/tutor/*`, and `/api/tutor/*` response includes `Cache-Control: no-store`.
- **Backend-Only Credentials**: `GEMINI_API_KEY` stays strictly on the backend via `SecretStr` and is never exposed to clients.
- **Zero Startup/Health Model Calls**: Starting the server or calling `GET /health` never invokes the Gemini API.

## Local Setup Instructions

### 1. Create and Activate Local Virtual Environment

```bash
python3 -m venv .venv
```

### 2. Install Reproducible Dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and set your backend `GEMINI_API_KEY`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
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
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`

### 5. Run Tests

```bash
.venv/bin/pytest
```
