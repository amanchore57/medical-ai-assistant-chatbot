# Medical Domain AI Assistant

A full-stack, domain-restricted AI assistant designed to deliver technical definitions, biological mechanisms, and etiology using open-weight models. Built with a FastAPI backend, real-time streaming over Server-Sent Events, and an interactive markdown-enabled web client.

---

## Features

- **Real-Time Token Streaming:** Delivers live token-by-token streaming using FastAPI's `StreamingResponse` and the native Web Streams API.
- **Medical Domain Guardrails:** Enforces domain-specific system prompt boundaries:
  - Answers queries strictly concerning human biology, medicine, and healthcare.
  - Politely refuses off-topic queries (e.g., coding, general trivia, math).
  - Enforces a declarative-only boundary (explains concepts/mechanisms while refusing step-by-step surgical instructions, dosage calculations, or self-treatment directions).
- **Deterministic Inference:** Locked at `temperature: 0.1` and `max_tokens: 700` to prevent repetition loops and enforce strict guardrail adherence.
- **Sliding Context Window:** Implements active memory slicing capping history to a 5-turn sliding buffer while locking the system prompt in place to avoid token overflow.
- **Session Persistence:** Saves session history to local JSON storage (`web_chat_history.json`), allowing conversations to survive server restarts and browser reloads.
- **Dynamic Markdown Rendering:** Renders Markdown tables, lists, and formatting dynamically in the browser using `marked.js`.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic, OpenAI Python SDK
- **Model Provider:** Hugging Face Inference Providers Router (`openai/gpt-oss-20b`)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Marked.js
- **Environment Management:** Python `dotenv`

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Hugging Face Access Token with serverless inference permissions

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_access_token_here
```

### 3. Installation
Activate your virtual environment and install the required dependencies:
```bash
pip install fastapi uvicorn openai python-dotenv rich
```

### 4. Running the Application

**Web Interface:**
Start the FastAPI server with auto-reload:
```bash
uvicorn main:app --reload
```
Access the application at `[http://127.0.0.1:8000](http://127.0.0.1:8000)`.

**CLI Interface (Alternative):**
To run the terminal client directly:
```bash
python app.py
```