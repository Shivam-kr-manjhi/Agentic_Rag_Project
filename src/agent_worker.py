"""
Agent Worker — semantic tool selection layer.

Receives a user query, embeds it, and finds the top-K most relevant
tools by comparing the query embedding against the persisted tool
description embeddings in ChromaDB. Acts as a routing intelligence
layer that narrows the search space before document-level retrieval.
"""

from typing import Dict, List

import chromadb

from src.config import embed_model, CHROMA_DB_DIR, TOP_K_TOOLS
from src.tool_factory import Tool


class AgentWorker:
    """Routes queries to the most relevant document tools."""

    def __init__(self, tools: List[Tool]):
        self.tools: Dict[str, Tool] = {t.name: t for t in tools}
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self._td_collection = self.chroma_client.get_collection(
            name="tool_descriptions"
        )

    def select_tools(self, query: str, top_k: int = TOP_K_TOOLS) -> List[Tool]:
        """
        Embed the query and find the top-K tools whose descriptions
        are most semantically similar.

        Returns a list of Tool objects sorted by relevance.
        """
        query_embedding = embed_model.encode([query]).tolist()

        # Clamp top_k to total available tools
        available = self._td_collection.count()
        k = min(top_k, available)

        results = self._td_collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        selected: List[Tool] = []
        if results["ids"] and results["ids"][0]:
            for tool_name in results["ids"][0]:
                if tool_name in self.tools:
                    selected.append(self.tools[tool_name])

        # Log selection
        print(f"\n[Worker] Query: \"{query}\"")
        print(f"[Worker] Selected {len(selected)} tool(s):")
        for i, t in enumerate(selected, 1):
            print(f"  {i}. {t.name} ({t.tool_type}) — {t.document_name}")

        return selected
