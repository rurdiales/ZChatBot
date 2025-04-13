from typing import List, Dict, Any
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from src.config import EMBEDDING_MODEL, VECTORDB_PATH

class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL, persist_dir: str = VECTORDB_PATH):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize or load existing vector store
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.db = Chroma(
                persist_directory=self.persist_dir, 
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector store with {self.db._collection.count()} documents")
        else:
            self.db = Chroma(
                persist_directory=self.persist_dir, 
                embedding_function=self.embeddings
            )
            print("Created new vector store")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            print("No documents to add")
            return
        
        self.db.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for documents most similar to the query."""
        results = self.db.similarity_search(query, k=k)
        return results
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Get all document IDs
        all_ids = self.db._collection.get()["ids"]
        if all_ids:
            # Delete all documents by their IDs
            self.db._collection.delete(ids=all_ids)
            print(f"Vector store cleared: {len(all_ids)} documents removed")
        else:
            print("Vector store is already empty") 