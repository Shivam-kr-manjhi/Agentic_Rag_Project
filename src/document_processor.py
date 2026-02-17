"""
Document Processor — reads, chunks, and persists documents to ChromaDB.

For each document in the Data/ folder:
  1. Extract full text (PDF via PyMuPDF, TXT/MD as-is)
  2. Chunk into overlapping segments
  3. Embed chunks with SentenceTransformer
  4. Store in a per-document ChromaDB collection
  5. Generate an LLM summary of the full document
"""

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import chromadb

from src.config import (
    embed_model,
    llm_chat,
    DATA_DIR,
    CHROMA_DB_DIR,
    MAX_CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
)


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class DocumentInfo:
    """Metadata for a processed document."""
    name: str               # original filename
    slug: str               # sanitized name for collection IDs
    summary: str            # LLM-generated summary
    chunk_count: int        # number of chunks stored
    collection_name: str    # ChromaDB collection name


# ── Helpers ─────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert filename to a safe collection slug."""
    stem = Path(name).stem
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    # ChromaDB collection names must be 3-63 chars and start/end with alphanumeric
    if len(slug) < 3:
        slug = slug + "_doc"
    return slug[:63]


def _extract_text(file_path: Path) -> str:
    """Extract text from a file (PDF, TXT, or MD)."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(str(file_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif suffix in (".txt", ".md", ".csv"):
        return file_path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _chunk_text(text: str, chunk_size: int = MAX_CHUNK_CHARS,
                overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks if chunks else [text.strip() or "(empty document)"]


def _file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    h = hashlib.md5()
    h.update(file_path.read_bytes())
    return h.hexdigest()


# ── Main Processor ──────────────────────────────────────────────────

class DocumentProcessor:
    """Process all documents in DATA_DIR and persist to ChromaDB."""

    def __init__(self):
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        # Metadata collection tracks which files have been processed
        self._meta_collection = self.chroma_client.get_or_create_collection(
            name="__document_meta__"
        )

    def process_all(self) -> List[DocumentInfo]:
        """Process every supported file in DATA_DIR. Returns DocumentInfo list."""
        supported = {".pdf", ".txt", ".md", ".csv"}
        files = [f for f in DATA_DIR.iterdir()
                 if f.is_file() and f.suffix.lower() in supported]

        if not files:
            print(f"[Processor] No supported documents found in {DATA_DIR}")
            return []

        results: List[DocumentInfo] = []
        for file_path in sorted(files):
            info = self._process_one(file_path)
            results.append(info)

        print(f"\n[Processor] Processed {len(results)} document(s) total.\n")
        return results

    # ── Per-document processing ─────────────────────────────────────

    def _process_one(self, file_path: Path) -> DocumentInfo:
        slug = _slugify(file_path.name)
        collection_name = f"doc_{slug}"
        current_hash = _file_hash(file_path)

        # Check if already processed and unchanged
        existing = self._meta_collection.get(ids=[slug])
        if existing and existing["documents"] and len(existing["documents"]) > 0:
            stored_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            if stored_meta.get("file_hash") == current_hash:
                print(f"[Processor] ✓ '{file_path.name}' already processed — skipping.")
                return DocumentInfo(
                    name=file_path.name,
                    slug=slug,
                    summary=existing["documents"][0],
                    chunk_count=int(stored_meta.get("chunk_count", 0)),
                    collection_name=collection_name,
                )

        # ── Full processing pipeline ────────────────────────────────
        print(f"[Processor] Processing '{file_path.name}' ...")

        # 1. Extract text
        full_text = _extract_text(file_path)
        print(f"  → Extracted {len(full_text)} characters")

        # 2. Chunk
        chunks = _chunk_text(full_text)
        print(f"  → Created {len(chunks)} chunks")

        # 3. Embed chunks
        print(f"  → Generating embeddings ...")
        embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()

        # 4. Store in ChromaDB (per-document collection)
        # Delete old collection if it exists, then recreate
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = self.chroma_client.create_collection(name=collection_name)

        ids = [f"{slug}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  → Stored {len(chunks)} chunks in collection '{collection_name}'")

        # 5. Generate summary via LLM
        print(f"  → Generating document summary ...")
        summary_prompt = (
            "You are a document analyst. Provide a comprehensive summary of the "
            "following document. Cover all major topics, themes, and key information.\n\n"
            f"DOCUMENT TEXT (first 6000 chars):\n{full_text[:6000]}\n\n"
            "SUMMARY:"
        )
        summary = llm_chat(summary_prompt)

        # 6. Store metadata
        self._meta_collection.upsert(
            ids=[slug],
            documents=[summary],
            metadatas=[{
                "file_name": file_path.name,
                "file_hash": current_hash,
                "chunk_count": str(len(chunks)),
                "collection_name": collection_name,
            }],
        )
        print(f"  ✓ Done processing '{file_path.name}'\n")

        return DocumentInfo(
            name=file_path.name,
            slug=slug,
            summary=summary,
            chunk_count=len(chunks),
            collection_name=collection_name,
        )
