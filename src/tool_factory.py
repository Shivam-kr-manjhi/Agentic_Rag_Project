"""
Tool Factory — creates vector, summary, and pandas analysis tools.

For each DocumentInfo:
  - If tabular (CSV/Excel): Creates a Pandas Analysis Tool
  - If text (PDF/TXT): Creates Vector + Summary Tools

Pandas Analysis Tool:
  - Generates Python/Pandas code via LLM to answer queries about the data
  - Executes code safely on the loaded dataframe
"""

import sys
import io
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd
import chromadb

from src.config import (
    embed_model,
    llm_chat,
    CHROMA_DB_DIR,
    TOP_K_CHUNKS,
)
from src.document_processor import DocumentInfo


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class Tool:
    """A callable tool with semantic metadata."""
    name: str
    description: str
    tool_type: str                   # "vector", "summary", or "pandas"
    document_name: str               # source document
    function: Callable[[str], str]   # accepts query, returns text


# ── Tool Factory ────────────────────────────────────────────────────

class ToolFactory:
    """Build tools for a set of documents."""

    def __init__(self):
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    def build_tools(self, doc_infos: List[DocumentInfo]) -> List[Tool]:
        """Create appropriate tools per document and persist descriptions."""
        tools: List[Tool] = []

        for info in doc_infos:
            if info.is_tabular:
                # ── Pandas Tool ─────────────────────────────────────
                pandas_tool = self._make_pandas_tool(info)
                tools.append(pandas_tool)
            else:
                # ── Vector Tool ─────────────────────────────────────
                vector_tool = self._make_vector_tool(info)
                tools.append(vector_tool)

                # ── Summary Tool ────────────────────────────────────
                summary_tool = self._make_summary_tool(info)
                tools.append(summary_tool)

        # ── Persist tool descriptions in ChromaDB ───────────────────
        self._persist_tool_descriptions(tools)

        print(f"[ToolFactory] Built {len(tools)} tools for {len(doc_infos)} document(s)\n")
        return tools

    # ── Pandas Tool Builder ─────────────────────────────────────────

    def _make_pandas_tool(self, info: DocumentInfo) -> Tool:
        name = f"pandas_analysis_{info.slug}"
        summary_snippet = info.summary[:200].replace("\n", " ")
        # A clearer description to help the agent select this tool for data questions
        description = (
            f"Analyze data in the file '{info.name}' using pandas. "
            f"Useful for counting, filtering, aggregating, finding averages, "
            f"sums, max/min values, or any structured data queries. "
            f"The dataset contains: {summary_snippet}..."
        )
        
        # Pre-load dataframe to avoid reading from disk on every query
        # (For very large files, you might want to load on demand instead)
        try:
            if info.name.endswith(".csv"):
                df = pd.read_csv(info.file_path)
            else:
                df = pd.read_excel(info.file_path)
        except Exception as e:
            print(f"[ToolFactory] Error loading dataframe for {info.name}: {e}")
            df = pd.DataFrame() # Fallback empty DF

        def pandas_fn(query: str, _df=df, _fname=info.name) -> str:
            """Generate and execute pandas code to answer the query."""
            
            # 1. Generate Code
            schema_info = []
            for col, dtype in _df.dtypes.items():
                schema_info.append(f"{col} ({dtype})")
            schema_str = ", ".join(schema_info)
            
            prompt = (
                f"You are a pandas data analysis assistant. \n"
                f"I have a dataframe named `df` from file '{_fname}'.\n"
                f"Columns: {schema_str}\n\n"
                f"USER QUERY: {query}\n\n"
                f"Write Python code to answer this query. \n"
                f"RULES:\n"
                f"1. Assume `df` is already loaded.\n"
                f"2. Store the final result in a variable named `result`.\n"
                f"3. Do NOT use print() — just assign `result`.\n"
                f"4. Return ONLY valid Python code, no markdown, no comments.\n"
                f"5. If the query asks for a plot, just return a string saying 'Plotting not supported yet'.\n"
            )
            
            code = llm_chat(prompt, system_prompt="You are a python coding machine. Output ONLY code.").strip()
            
            # Sanitization (basic)
            code = code.replace("```python", "").replace("```", "").strip()
            
            print(f"\n[PandasTool] Generated code:\n{code}\n")
            
            # 2. Execute Code
            local_vars = {"df": _df, "result": None}
            
            # Redirect stdout to capture any print usage if the LLM violates the rule
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                exec(code, {}, local_vars)
                result = local_vars.get("result")
                sys.stdout = old_stdout # Restore stdout
                
                output_str = captured_output.getvalue().strip()
                
                if result is not None:
                    return str(result)
                elif output_str:
                    return output_str
                else:
                    return "Code executed successfully but `result` variable was None."
                    
            except Exception as e:
                sys.stdout = old_stdout # Restore stdout
                return f"Error executing pandas code:\n{traceback.format_exc()}"

        return Tool(
            name=name,
            description=description,
            tool_type="pandas",
            document_name=info.name,
            function=pandas_fn,
        )

    # ── Vector Tool Builder ─────────────────────────────────────────

    def _make_vector_tool(self, info: DocumentInfo) -> Tool:
        collection_name = info.collection_name
        summary_snippet = info.summary[:200].replace("\n", " ")

        name = f"vector_search_{info.slug}"
        description = (
            f"Search for specific facts, figures, names, dates, definitions, "
            f"or detailed information within the document '{info.name}'. "
            f"This document covers: {summary_snippet}..."
        )

        def vector_fn(query: str, _cname=collection_name) -> str:
            """Query ChromaDB collection for top-K similar chunks."""
            collection = self.chroma_client.get_collection(name=_cname)
            query_embedding = embed_model.encode([query]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=TOP_K_CHUNKS,
            )
            if results["documents"] and results["documents"][0]:
                chunks = results["documents"][0]
                return "\n\n---\n\n".join(
                    f"[Chunk {i+1}] {chunk}"
                    for i, chunk in enumerate(chunks)
                )
            return "(No relevant chunks found)"

        return Tool(
            name=name,
            description=description,
            tool_type="vector",
            document_name=info.name,
            function=vector_fn,
        )

    # ── Summary Tool Builder ────────────────────────────────────────

    def _make_summary_tool(self, info: DocumentInfo) -> Tool:
        summary_text = info.summary
        summary_snippet = info.summary[:200].replace("\n", " ")

        name = f"summary_{info.slug}"
        description = (
            f"Get a high-level overview, themes, main topics, or general "
            f"understanding of the document '{info.name}'. "
            f"This document covers: {summary_snippet}..."
        )

        def summary_fn(query: str, _summary=summary_text) -> str:
            return _summary

        return Tool(
            name=name,
            description=description,
            tool_type="summary",
            document_name=info.name,
            function=summary_fn,
        )

    # ── Tool Description Persistence ────────────────────────────────

    def _persist_tool_descriptions(self, tools: List[Tool]) -> None:
        """Embed and store all tool descriptions in ChromaDB for semantic lookup."""
        # Use get_or_create + upsert to avoid HNSW index flush race conditions
        # that cause intermittent "Nothing found on disk" errors
        td_collection = self.chroma_client.get_or_create_collection(
            name="tool-descriptions"
        )

        names = [t.name for t in tools]
        descriptions = [t.description for t in tools]
        embeddings = embed_model.encode(descriptions, show_progress_bar=False).tolist()
        metadatas = [
            {"tool_type": t.tool_type, "document_name": t.document_name}
            for t in tools
        ]

        td_collection.upsert(
            ids=names,
            documents=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[ToolFactory] Persisted {len(tools)} tool descriptions to ChromaDB")
