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
import logging

# Redirect stdout/stderr to file for debugging
# sys.stdout = open("server.log", "w", encoding="utf-8")
# sys.stderr = sys.stdout

# Configure logging
logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Also add a stream handler to print to console so we don't lose it entirely if needed
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


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

def get_bot_reply(message, docs=None):
    query = message
    answer = runner.run(query, allowed_docs=docs)
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

    # If docs is empty list, treat as None (all docs) or empty? 
    # Usually empty selection means "all" in some UIs, or "none" in others. 
    # Based on user request "only provide documents tool amoungst tehm only", 
    # if the user selects nothing, maybe they want to search everything?
    # Let's assume if docs is empty, we pass None to let the agent decide (which currently defaults to all).
    # However, if the UI explicitly sends [], it might mean "no constraints".
    
    reply = get_bot_reply(message, docs=docs if docs else None)
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