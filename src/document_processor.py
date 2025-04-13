import os
import glob
from typing import List, Dict, Any
import docx2txt
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain.schema import Document
from src.config import KNOWLEDGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    """Process different document types and convert them to text chunks."""
    
    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    
    def process_all_documents(self) -> List[Document]:
        """Process all documents in the knowledge directory and return chunks."""
        all_docs = []
        
        # Process each supported file type
        for ext, processor in {
            "**/*.pdf": self._process_pdf,
            "**/*.docx": self._process_docx,
            "**/*.txt": self._process_text,
            "**/*.png": self._process_image,
            "**/*.jpg": self._process_image,
            "**/*.jpeg": self._process_image,
        }.items():
            pattern = os.path.join(self.knowledge_dir, ext)
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    docs = processor(file_path)
                    all_docs.extend(docs)
                    print(f"Processed {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        return all_docs
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def _process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX files."""
        text = docx2txt.process(file_path)
        metadata = {"source": file_path}
        documents = [{"page_content": text, "metadata": metadata}]
        return self.text_splitter.create_documents([text], [metadata])
    
    def _process_text(self, file_path: str) -> List[Document]:
        """Process text files."""
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def _process_image(self, file_path: str) -> List[Document]:
        """Process image files using OCR."""
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        metadata = {"source": file_path}
        return self.text_splitter.create_documents([text], [metadata])
    
    def _process_generic(self, file_path: str) -> List[Document]:
        """Process other file types using Unstructured."""
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks 