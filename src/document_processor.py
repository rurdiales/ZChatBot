import os
import glob
import re
from typing import List, Dict, Any, Optional
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

# Import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available, install with: pip install PyMuPDF")

# Import language detection libraries
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not available, install with: pip install langdetect")

# Set minimum text threshold for considering a PDF successfully processed
# If text extracted is less than this percentage of expected text, use OCR fallback
MIN_TEXT_THRESHOLD = 100  # Minimum characters expected from a valid PDF page

# Map of language codes: langdetect format to Tesseract format
LANG_CODE_MAP = {
    'es': 'spa',  # Spanish
    'en': 'eng',  # English
    'fr': 'fra',  # French
    'de': 'deu',  # German
    'it': 'ita',  # Italian
    'pt': 'por',  # Portuguese
    'nl': 'nld',  # Dutch
    'ru': 'rus',  # Russian
    'zh-cn': 'chi_sim',  # Chinese (Simplified)
    'zh-tw': 'chi_tra',  # Chinese (Traditional)
    'ja': 'jpn',  # Japanese
    'ko': 'kor',  # Korean
    'ar': 'ara',  # Arabic
}

# Default language to use if detection fails
DEFAULT_OCR_LANG = 'spa'  # Spanish is the default for this user

class DocumentProcessor:
    """Process different document types and convert them to text chunks."""
    
    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        # Cache detected languages for files
        self.detected_languages = {}
    
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
        """Process PDF files using PyMuPDF with smart Tesseract OCR fallback."""
        if PYMUPDF_AVAILABLE:
            try:
                # First try with PyMuPDF for fastest and best extraction
                text_by_page = self._process_pdf_with_pymupdf(file_path)
                
                # Better heuristics for OCR decision:
                # 1. Calculate average text per page for context
                total_pages = len(text_by_page)
                if total_pages == 0:
                    return self._process_pdf_with_ocr(file_path)
                    
                total_text = sum(len(text.strip()) for text in text_by_page.values())
                avg_chars_per_page = total_text / total_pages
                
                print(f"PDF {file_path}: {total_pages} pages, avg {avg_chars_per_page:.1f} chars/page")
                
                # Try to detect language from extracted text if sufficient
                if avg_chars_per_page > 200:
                    # Get a sample of text from the middle of the document
                    sample_pages = [text_by_page[p] for p in sorted(text_by_page.keys())[:min(3, total_pages)]]
                    sample_text = "\n".join(sample_pages)
                    detected_lang = self._detect_language(sample_text)
                    if detected_lang:
                        self.detected_languages[file_path] = detected_lang
                        print(f"Detected language: {detected_lang}")
                
                # 2. Count low-text pages
                low_text_pages = []
                for page_num, page_text in text_by_page.items():
                    # Skip common low-text pages (first and last often have less text)
                    if page_num == 0 or page_num == total_pages - 1:
                        continue
                        
                    # Page has significantly less text than average (less than 15%)
                    page_text_len = len(page_text.strip())
                    relative_threshold = max(MIN_TEXT_THRESHOLD, avg_chars_per_page * 0.15)
                    
                    if page_text_len < relative_threshold:
                        low_text_pages.append((page_num, page_text_len))
                
                # Only use OCR if multiple pages have suspicious text levels
                # or if total document text density is very low
                needs_ocr = (len(low_text_pages) > max(1, total_pages * 0.25) or 
                            (avg_chars_per_page < 200 and total_pages > 1))
                
                if needs_ocr:
                    # Log the reason for OCR
                    if len(low_text_pages) > 0:
                        print(f"Using OCR fallback: {len(low_text_pages)} suspicious pages detected:")
                        for page_num, char_count in low_text_pages:
                            print(f"  - Page {page_num+1}: only {char_count} chars (threshold: {max(MIN_TEXT_THRESHOLD, avg_chars_per_page * 0.15):.1f})")
                    else:
                        print(f"Using OCR fallback: Low overall text density ({avg_chars_per_page:.1f} chars/page)")
                    
                    return self._process_pdf_with_ocr(file_path)
                else:
                    # Create Document objects for each page
                    documents = []
                    for page_num, page_text in text_by_page.items():
                        documents.append(Document(
                            page_content=page_text,
                            metadata={"source": file_path, "page": page_num}
                        ))
                    chunks = self.text_splitter.split_documents(documents)
                    print(f"Successfully processed PDF with PyMuPDF: {file_path}")
                    return chunks
            except Exception as e:
                print(f"PyMuPDF failed for {file_path}, falling back to OCR: {str(e)}")
                return self._process_pdf_with_ocr(file_path)
        else:
            # Fall back to PyPDFLoader if PyMuPDF is not available
            print(f"PyMuPDF not available, using PyPDFLoader for {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return chunks
    
    def _process_pdf_with_pymupdf(self, file_path: str) -> Dict[int, str]:
        """Extract text from PDF using PyMuPDF (fitz)."""
        text_by_page = {}
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            text_by_page[page_num] = text
        doc.close()
        return text_by_page
    
    def _process_pdf_with_ocr(self, file_path: str) -> List[Document]:
        """Process PDF with Tesseract OCR by converting to images."""
        from pdf2image import convert_from_path
        
        try:
            print(f"Converting PDF to images for OCR: {file_path}")
            
            # First convert just the first page to detect language if needed
            images_first_page = convert_from_path(file_path, first_page=1, last_page=1)
            
            # Detect language if not already detected
            if file_path not in self.detected_languages and images_first_page:
                # Run initial OCR with multi-language support to detect language
                initial_text = pytesseract.image_to_string(images_first_page[0], lang='eng+spa+fra+deu')
                if initial_text and len(initial_text.strip()) > 50:  # If we got reasonable text
                    detected_lang = self._detect_language(initial_text)
                    if detected_lang:
                        self.detected_languages[file_path] = detected_lang
                        print(f"Auto-detected language for OCR: {detected_lang}")
            
            # Get the language to use
            tesseract_lang = self._get_tesseract_lang(file_path)
            print(f"Using OCR language: {tesseract_lang}")
            
            # Now process all pages with the detected language
            images = convert_from_path(file_path)
            
            documents = []
            for i, image in enumerate(images):
                # Extract text with Tesseract using detected language
                text = pytesseract.image_to_string(image, lang=tesseract_lang)
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": i, "ocr": True, "language": tesseract_lang}
                ))
            
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            print(f"OCR processing failed for {file_path}: {str(e)}")
            # Last resort - use PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return chunks
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language from text and return Tesseract language code."""
        if not LANGDETECT_AVAILABLE or not text:
            return None
            
        try:
            # Clean text for better detection
            # Remove common non-linguistic patterns that confuse detection
            cleaned_text = re.sub(r'\d+', '', text)  # Remove numbers
            cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation
            
            if len(cleaned_text.strip()) < 50:
                return None
                
            # Detect language
            lang_code = detect(cleaned_text)
            
            # Map to Tesseract language code
            tesseract_lang = LANG_CODE_MAP.get(lang_code)
            if tesseract_lang:
                return tesseract_lang
            else:
                print(f"Detected language '{lang_code}' not in mapping, using default")
                return DEFAULT_OCR_LANG
        except LangDetectException as e:
            print(f"Language detection failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in language detection: {e}")
            return None
    
    def _get_tesseract_lang(self, file_path: str) -> str:
        """Get the appropriate Tesseract language for the file."""
        # First check if we've already detected the language
        if file_path in self.detected_languages:
            return self.detected_languages[file_path]
            
        # If no language detected, use default
        return DEFAULT_OCR_LANG
    
    def _process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX files."""
        text = docx2txt.process(file_path)
        metadata = {"source": file_path}
        return self.text_splitter.create_documents([text], [metadata])
    
    def _process_text(self, file_path: str) -> List[Document]:
        """Process text files."""
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def _process_image(self, file_path: str) -> List[Document]:
        """Process image files using Tesseract OCR."""
        try:
            image = Image.open(file_path)
            
            # First try to detect language if not already detected
            if file_path not in self.detected_languages:
                # Try initial OCR with multi-language support
                initial_text = pytesseract.image_to_string(image, lang='eng+spa+fra+deu')
                if initial_text:
                    detected_lang = self._detect_language(initial_text)
                    if detected_lang:
                        self.detected_languages[file_path] = detected_lang
                        print(f"Auto-detected language for image OCR: {detected_lang}")
            
            # Get the language to use
            tesseract_lang = self._get_tesseract_lang(file_path)
            print(f"Using OCR language for image: {tesseract_lang}")
            
            # Process with detected language
            text = pytesseract.image_to_string(image, lang=tesseract_lang)
            
            metadata = {"source": file_path, "language": tesseract_lang}
            return self.text_splitter.create_documents([text], [metadata])
        except Exception as e:
            print(f"Image processing failed for {file_path}: {str(e)}")
            return []
            
    def _process_image_with_pytesseract(self, image: Image.Image) -> str:
        """Process an image using pytesseract with language auto-detection."""
        # Get a file path key for this image (memory address as unique identifier)
        image_id = str(id(image))
        
        # If language not detected yet for this image
        if image_id not in self.detected_languages:
            # Try initial OCR with multi-language support
            try:
                initial_text = pytesseract.image_to_string(image, lang='eng+spa+fra+deu')
                if initial_text:
                    detected_lang = self._detect_language(initial_text)
                    if detected_lang:
                        self.detected_languages[image_id] = detected_lang
            except Exception:
                # If multi-language detection fails, default to DEFAULT_OCR_LANG
                pass
        
        # Get the language to use (or default)
        tesseract_lang = self.detected_languages.get(image_id, DEFAULT_OCR_LANG)
        
        try:
            # Try with detected/default language
            return pytesseract.image_to_string(image, lang=tesseract_lang)
        except Exception as e:
            print(f"Error using language '{tesseract_lang}' with Tesseract: {e}")
            # Fall back to default Tesseract language (usually 'eng')
            return pytesseract.image_to_string(image)
    
    def _process_generic(self, file_path: str) -> List[Document]:
        """Process other file types using Unstructured."""
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks 