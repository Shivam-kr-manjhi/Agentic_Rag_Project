
from src.document_processor import DocumentProcessor
from src.tool_factory import ToolFactory
from src.agent_worker import AgentWorker

def test_filtering():
    print("1. Processing documents...")
    processor = DocumentProcessor()
    doc_infos = processor.process_all()
    
    print("2. Building tools...")
    factory = ToolFactory()
    tools = factory.build_tools(doc_infos)
    
    print("3. Initializing Worker...")
    worker = AgentWorker(tools)
    
    query = "What is the average salary in the Engineering department?"
    
    print("\n--- TEST CASE 1: No filter (should find employees.csv tool) ---")
    selected = worker.select_tools(query)
    print(f"Selected: {[t.name for t in selected]}")
    
    print("\n--- TEST CASE 2: Filter to 'employees.csv' (should find it) ---")
    selected = worker.select_tools(query, allowed_docs=["employees.csv"])
    print(f"Selected: {[t.name for t in selected]}")
    
    print("\n--- TEST CASE 3: Filter to 'irrelevant.pdf' (should NOT find it) ---")
    # Use a doc name that exists but is irrelevant
    irrelevant_doc = "5-6th sem marksheets _compressed.pdf" 
    selected = worker.select_tools(query, allowed_docs=[irrelevant_doc])
    print(f"Selected: {[t.name for t in selected]}")

if __name__ == "__main__":
    test_filtering()
