"""
Configuration module for the Agentic RAG system.
Loads environment variables, initializes the Groq LLM client and
SentenceTransformer embedding model, and defines project-wide paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Load environment variables ──────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.environ.get("Groq_KEY", "")
HUGGINGFACE_KEY = os.environ.get("HuggingFace_Key", "")

if not GROQ_API_KEY:
    raise EnvironmentError("Groq_KEY not found in .env file")

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "uploads"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ── LLM Client (Groq) ──────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0

groq_client = Groq(api_key=GROQ_API_KEY)


def llm_chat(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Send a prompt to the Groq LLM and return the response text."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content


# ── Embedding Model (local SentenceTransformer) ────────────────────
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

print(f"[Config] Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"[Config] Embedding model loaded (dim={embed_model.get_sentence_embedding_dimension()})")

# ── Chunking Parameters ────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens (approx chars / 4)
CHUNK_OVERLAP = 50        # token overlap between chunks
MAX_CHUNK_CHARS = 2048    # character-based chunk size
CHUNK_OVERLAP_CHARS = 200 # character-based overlap

# ── Agent Parameters ───────────────────────────────────────────────
TOP_K_TOOLS = 3           # number of tools selected by the agent worker
TOP_K_CHUNKS = 5          # number of chunks retrieved per vector search
MAX_REASONING_STEPS = 3   # max iterations in the reasoning loop
