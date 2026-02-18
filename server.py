import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
"""
Main entry point — Interactive CLI for the Agentic RAG system.

Workflow:
  1. Process all documents in Data/ (skip if already persisted in ChromaDB)
  2. Build vector + summary tools via ToolFactory
  3. Initialize AgentWorker (semantic tool selection) and AgentRunner (reasoning)
  4. Enter interactive query loop
"""

import sys
from src.document_processor import DocumentProcessor
from src.tool_factory import ToolFactory
from src.agent_worker import AgentWorker
from src.agent_runner import AgentRunner

app = Flask(__name__)
CORS(app)

CHATS_DIR = os.path.join(os.getcwd(), "chats")
UPLOADS_DIR = os.path.join(os.getcwd(), "uploads")

os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── Step 1: Document Processing ─────────────────────────────────
processor = DocumentProcessor()
doc_infos = processor.process_all()

# ── Step 2: Tool Construction ───────────────────────────────────
factory = ToolFactory()
tools = factory.build_tools(doc_infos)

# ── Step 3: Agent Initialization ────────────────────────────────
worker = AgentWorker(tools)
runner = AgentRunner(worker)


# ─── Hardcoded AI reply ───────────────────────────────────────────────────────

def get_bot_reply(message):
    query = message
    answer = runner.run(query)
    return f"{answer}"

# ─── POST /chat ───────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    message = body.get("message", "")
    docs = body.get("docs", [])  # list of enabled doc filenames

    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    # docs contains the checked filenames — use them when wiring real AI
    print(f"[chat] message: {message}")
    print(f"[chat] active docs: {docs}")

    reply = get_bot_reply(message)
    return jsonify({"reply": reply, "docs_used": docs})


# ─── GET /chats ───────────────────────────────────────────────────────────────

@app.route("/chats", methods=["GET"])
def get_chats():
    files = os.listdir(CHATS_DIR)
    chats = []

    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(CHATS_DIR, file), "r") as f:
                chats.append(json.load(f))

    return jsonify(chats)


# ─── POST /chats ──────────────────────────────────────────────────────────────

@app.route("/chats", methods=["POST"])
def save_chat():
    body = request.get_json()

    if not body or "id" not in body:
        return jsonify({"error": "Invalid chat data"}), 400

    file_name = f"chat_{body['id']}.json"
    file_path = os.path.join(CHATS_DIR, file_name)

    with open(file_path, "w") as f:
        json.dump(body, f, indent=2, default=str)

    return jsonify({"success": True})


# ─── POST /upload ─────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "No files received"}), 400

    saved = []
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOADS_DIR, filename))
        saved.append(filename)

    return jsonify({"uploaded": saved})


# ─── GET /upload ──────────────────────────────────────────────────────────────

@app.route("/upload", methods=["GET"])
def list_uploads():
    files = os.listdir(UPLOADS_DIR)
    return jsonify({"files": files})


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(port=8000, debug=True)