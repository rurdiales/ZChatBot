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
            
    def vacuum_database(self) -> None:
        """Vacuum the ChromaDB database to optimize after version upgrades."""
        try:
            # Access the underlying collection client
            print("Vacuuming ChromaDB database...")
            
            # Get the underlying Chroma client
            client = self.db._collection._client
            
            # Different ways to try vacuuming depending on ChromaDB version
            try:
                # Newer ChromaDB versions
                client.vacuum()
                print("Database vacuum completed successfully using client.vacuum()")
            except (AttributeError, TypeError) as e1:
                try:
                    # Try alternative method
                    client._system.vacuum()
                    print("Database vacuum completed successfully using client._system.vacuum()")
                except (AttributeError, TypeError) as e2:
                    try:
                        # Try direct access to the SQLite database
                        import sqlite3
                        db_path = os.path.join(self.persist_dir, "chroma.sqlite3")
                        if os.path.exists(db_path):
                            conn = sqlite3.connect(db_path)
                            conn.execute("VACUUM")
                            conn.close()
                            print("Database vacuum completed successfully using direct SQLite VACUUM")
                        else:
                            raise FileNotFoundError(f"SQLite database not found at {db_path}")
                    except Exception as e3:
                        raise Exception(f"Failed all vacuum methods: {e1}, {e2}, {e3}")
            
        except Exception as e:
            print(f"Error during database vacuum: {e}")
            print("If the error persists, you may need to rebuild the vector database:")
            print("1. Backup your documents in the knowledge folder")
            print("2. Delete the data/vectordb directory")
            print("3. Run the chatbot with --process to recreate the database") 