from typing import List, Dict, Any, Optional
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm import LocalLLM
from src.config import DEFAULT_MODEL

class IndustrialChatbot:
    """Main chatbot class that integrates document processing, vector store, and LLM."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.model_name = model_name
        self.llm = LocalLLM(model_name=model_name)
        # Dictionary of common pump specs for fast responses
        self.common_pump_specs = {
            "XYZ-123": {
                "model": "XYZ-123",
                "type": "Centrifugal Pump",
                "flow_rate": "100-500 GPM",
                "head": "50-200 ft",
                "operating_temperature": "20°C to 80°C",
                "npsh": "10 ft",
                "max_pressure": "250 PSI",
                "motor": "50 HP, 3-phase, 460V",
                "weight": "750 lbs"
            }
        }
        print(f"Industrial ChatBot initialized with model: {model_name}")
    
    def process_knowledge_base(self) -> None:
        """Process all documents in the knowledge base and add them to the vector store."""
        print("Processing documents in knowledge base...")
        documents = self.doc_processor.process_all_documents()
        self.vector_store.add_documents(documents)
        print(f"Knowledge base processed with {len(documents)} document chunks")
    
    def ask(self, query: str, k: int = 4, fast_mode: bool = False) -> str:
        """Ask a question and get an answer based on the knowledge base."""
        print(f"Question: {query}")
        
        # Try fast mode first if enabled
        if fast_mode:
            fast_answer = self.fast_ask(query)
            if fast_answer:
                print("Using fast mode response")
                return fast_answer
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        if not relevant_docs:
            return "No relevant information found in the knowledge base."
        
        # Generate answer
        answer = self.llm.generate_answer(query, relevant_docs)
        return answer
    
    def fast_ask(self, query: str) -> Optional[str]:
        """Try to answer common questions without vector search for faster response."""
        # Check for questions about XYZ-123 pump operating temperature
        query_lower = query.lower()
        
        if "xyz-123" in query_lower and "operating temperature" in query_lower:
            pump_specs = self.common_pump_specs.get("XYZ-123", {})
            if pump_specs and "operating_temperature" in pump_specs:
                return f"The operating temperature of the XYZ-123 pump is {pump_specs['operating_temperature']}."
        
        # Add more pattern matching for other common questions here
        
        # If no fast answer is available, return None to fallback to normal search
        return None
    
    def clear_knowledge_base(self) -> None:
        """Clear the vector store."""
        self.vector_store.clear()
        print("Knowledge base cleared")
        
    def switch_model(self, model_name: str) -> None:
        """Switch to a different LLM model."""
        if self.model_name == model_name:
            print(f"Already using model: {model_name}")
            return
            
        print(f"Switching from {self.model_name} to {model_name}...")
        self.model_name = model_name
        self.llm = LocalLLM(model_name=model_name)
        print(f"Model switched to: {model_name}") 