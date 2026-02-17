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


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║            🧠  AGENTIC RAG SYSTEM  🧠                       ║
║   Multi-Layer Document Intelligence with Iterative Reasoning ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── Step 1: Document Processing ─────────────────────────────────
    print("=" * 60)
    print("  PHASE 1: DOCUMENT PROCESSING & INDEXING")
    print("=" * 60)
    processor = DocumentProcessor()
    doc_infos = processor.process_all()

    if not doc_infos:
        print("No documents found. Please add files to the Data/ folder.")
        sys.exit(1)

    for info in doc_infos:
        print(f"  📄 {info.name} — {info.chunk_count} chunks | collection: {info.collection_name}")
    print()

    # ── Step 2: Tool Construction ───────────────────────────────────
    print("=" * 60)
    print("  PHASE 2: BUILDING DOCUMENT TOOLS")
    print("=" * 60)
    factory = ToolFactory()
    tools = factory.build_tools(doc_infos)

    for tool in tools:
        print(f"  🔧 {tool.name} ({tool.tool_type})")
    print()

    # ── Step 3: Agent Initialization ────────────────────────────────
    print("=" * 60)
    print("  PHASE 3: AGENT INITIALIZATION")
    print("=" * 60)
    worker = AgentWorker(tools)
    runner = AgentRunner(worker)
    print("[Agent] Ready for queries!\n")

    # ── Step 4: Interactive Query Loop ──────────────────────────────
    print("=" * 60)
    print("  Type your question below (type 'quit' or 'exit' to stop)")
    print("=" * 60)

    while True:
        try:
            query = input("\n❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        print()
        answer = runner.run(query)

        print("\n" + "─" * 60)
        print("  💡 FINAL ANSWER")
        print("─" * 60)
        print(answer)
        print("─" * 60)


if __name__ == "__main__":
    main()
