"""
api.py — FastAPI backend for the AI Expense Chatbot.

Exposes REST endpoints for receipt upload/processing, expense tracking,
chat with the AI agent, and dashboard analytics.

NOTE: Endpoints use regular `def` (not `async def`) because the underlying
code (SQLite, EasyOCR, Ollama, LangGraph) is all synchronous/blocking.
FastAPI automatically runs `def` endpoints in a threadpool so they don't
block the event loop.
"""

import os
import uuid
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.processor import extract_text_from_image, parse_with_ollama
from backend.memory import (
    init_dbs,
    save_receipt_data,
    get_all_receipts,
    get_current_month_total,
    get_total_spend,
    reset_dbs,
)
from backend.agent import build_agent, run_agent

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(title="AI Expense Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize databases on startup
init_dbs()

# Build agent (global state)
agent = None
chat_history: list[dict] = []


def _get_agent():
    """Lazy-init the agent."""
    global agent
    if agent is None:
        agent = build_agent()
    return agent


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class ReceiptSaveRequest(BaseModel):
    merchant: str
    date: str
    total: float
    currency: str
    category: str
    raw_text: str = ""


class ChatRequest(BaseModel):
    message: str


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.post("/api/upload")
def upload_receipt(file: UploadFile = File(...)):
    """Upload a receipt image → OCR → LLM extraction.
    Returns the raw text and structured data (not saved yet).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Save to temp file for EasyOCR
    suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    tmp_path = os.path.join(tempfile.gettempdir(), f"receipt_{uuid.uuid4().hex}{suffix}")

    try:
        contents = file.file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        raw_text = extract_text_from_image(tmp_path)
        if not raw_text:
            return {
                "success": False,
                "error": "Could not read any text from the image. Try a clearer photo.",
                "raw_text": "",
                "data": None,
            }

        structured_data = parse_with_ollama(raw_text)
        return {
            "success": True,
            "raw_text": raw_text,
            "data": structured_data,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/save")
def save_receipt(receipt: ReceiptSaveRequest):
    """Save a confirmed receipt to both SQLite and ChromaDB."""
    data = {
        "merchant": receipt.merchant,
        "date": receipt.date,
        "total": receipt.total,
        "currency": receipt.currency,
        "category": receipt.category,
    }
    save_receipt_data(receipt.raw_text or "", data)
    return {"success": True, "message": f"Saved {receipt.currency}{receipt.total:.2f} at {receipt.merchant}"}


@app.get("/api/receipts")
def get_receipts():
    """Get all receipts as a list of dicts."""
    df = get_all_receipts()
    if df.empty:
        return {"receipts": [], "count": 0}
    return {"receipts": df.to_dict(orient="records"), "count": len(df)}


@app.get("/api/stats")
def get_stats():
    """Get dashboard statistics."""
    df = get_all_receipts()
    month_total = get_current_month_total()
    all_time = get_total_spend()

    category_breakdown = {}
    if not df.empty:
        category_breakdown = df.groupby("category")["total"].sum().to_dict()

    return {
        "month_total": round(month_total, 2),
        "all_time_total": round(all_time, 2),
        "receipt_count": len(df),
        "category_breakdown": category_breakdown,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Send a message to the AI agent and get a response."""
    global chat_history

    try:
        current_agent = _get_agent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {e}")

    response = run_agent(current_agent, req.message, chat_history)

    chat_history.append({"role": "user", "content": req.message})
    chat_history.append({"role": "assistant", "content": response})

    return {"response": response}


@app.delete("/api/chat/history")
def clear_chat():
    """Clear the chat history."""
    global chat_history
    chat_history = []
    return {"success": True, "message": "Chat history cleared."}


@app.post("/api/reset-agent")
def reset_agent():
    """Rebuild the AI agent."""
    global agent
    try:
        agent = build_agent()
        return {"success": True, "message": "Agent rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset-db")
def reset_database():
    """Wipe all databases and rebuild."""
    global chat_history
    reset_dbs()
    chat_history = []
    return {"success": True, "message": "Databases wiped and reinitialized."}


# ──────────────────────────────────────────────
# Serve Frontend
# ──────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")
