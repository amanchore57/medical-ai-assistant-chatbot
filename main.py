import os
import json
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

app = FastAPI(title="AI Assistant")

# --- GUARDRAILS & PARAMETERS ---
SYSTEM_PROMPT = (
    "You are an AI Assistant specialized in medical science and healthcare information. "
    "Strictly adhere to the following rules:\n\n"
    "1. DOMAIN SCOPE: Only answer questions related to human biology, healthcare, and medical science. "
    "If the user asks an off-topic question (e.g., programming, general trivia, travel, math, history), "
    "politely inform them that you are designed specifically to answer medical and health-related topics only.\n\n"
    "2. DESCRIPTIVE ONLY: Explain medical concepts, conditions, history, anatomy, and biological mechanisms clearly. "
    "Organize complex answers using clean Markdown formatting and tables where appropriate.\n\n"
    "3. NO PROCEDURAL GUIDANCE: Never provide step-by-step surgical instructions, execution guides, "
    "dosage calculations, or self-treatment directions. If asked how to perform a procedure or administer a treatment, "
    "explain what the treatment is and its general purpose, but state that you cannot provide execution instructions."
)

TEMPERATURE = 0.1
MAX_TOKENS = 700
MAX_TURNS = 5
HISTORY_FILE = "web_chat_history.json"


def load_sessions() -> Dict[str, List[dict]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_sessions(sessions_data: Dict[str, List[dict]]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(sessions_data, f, indent=4)


sessions: Dict[str, List[dict]] = load_sessions()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"


class ClearRequest(BaseModel):
    session_id: str = "default_session"


@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    if session_id in sessions:
        return {"history": sessions[session_id]}
    return {"history": []}


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    session_key = payload.session_id

    if session_key not in sessions:
        sessions[session_key] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    sessions[session_key].append({"role": "user", "content": payload.message})
    save_sessions(sessions)

    def token_stream_generator():
        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=sessions[session_key],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_reply += token
                yield token

        sessions[session_key].append({"role": "assistant", "content": full_reply})

        if len(sessions[session_key]) > 1 + (MAX_TURNS * 2):
            sessions[session_key] = (
                [sessions[session_key][0]] + sessions[session_key][-(MAX_TURNS * 2):]
            )

        save_sessions(sessions)

    return StreamingResponse(token_stream_generator(), media_type="text/plain")


@app.post("/api/clear")
def clear_chat_session(payload: ClearRequest):
    session_key = payload.session_id
    if session_key in sessions:
        sessions[session_key] = [{"role": "system", "content": SYSTEM_PROMPT}]
        save_sessions(sessions)
    return {"status": "cleared"}