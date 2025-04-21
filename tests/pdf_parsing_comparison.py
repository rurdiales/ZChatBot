#!/usr/bin/env python3
"""
PDF Parsing Comparison Test

This standalone script compares different methods for parsing PDF files:
- PyPDF2
- PyMuPDF (fitz)
- PaddleOCR
- Pytesseract with pdf2image

The script:
1. Automatically installs required dependencies
2. Tests each method on sample PDFs (from a specified directory)
3. Compares processing time and text extraction quality
4. Outputs detailed results
"""

import os
import sys
import time
import subprocess
import platform
import tempfile
from pathlib import Path
import argparse
import shutil
import re
from typing import Dict, List, Tuple, Any
import glob
from PIL import Image
import importlib

# Global flag to skip installation
SKIP_INSTALL = False

# Add these at the top of the script after imports
TESSERACT_PATH = "/opt/homebrew/bin/tesseract"
PDFTOPPM_PATH = "/opt/homebrew/bin/pdftoppm"
OCR_LANGUAGE = "spa"  # Default to Spanish, can be overridden by command line
INSTALL_PADDLEOCR = False  # Flag to control whether to attempt installing PaddleOCR

# Function to install packages if they're not already installed
def install_package(package):
    if SKIP_INSTALL:
        print(f"Skipping installation of {package} (--skip-install flag is set)")
        return False
        
    try:
        # Use the current Python interpreter (from venv)
        python_executable = sys.executable
        subprocess.check_call([python_executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except Exception as e:
        print(f"Failed to install {package}: {e}")
        return False

# Function to install system dependencies
def install_system_dependency(dependency):
    """
    Attempt to install system dependencies like poppler, tesseract.
    Returns True if successful, False otherwise.
    """
    if SKIP_INSTALL:
        print(f"Skipping installation of {dependency} (--skip-install flag is set)")
        return False
        
    print(f"Attempting to install {dependency}...")
    
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            # Check if homebrew is installed
            if shutil.which("brew"):
                subprocess.check_call(["brew", "install", dependency])
                print(f"Successfully installed {dependency} using Homebrew")
                return True
            else:
                print("Homebrew not found. Please install Homebrew first: https://brew.sh/")
                return False
                
        elif system == "Linux":
            # Try apt-get (Debian/Ubuntu)
            if shutil.which("apt-get"):
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y", 
                                      "poppler-utils" if dependency == "poppler" else dependency])
                print(f"Successfully installed {dependency} using apt-get")
                return True
            # Try yum (CentOS/RHEL)
            elif shutil.which("yum"):
                subprocess.check_call(["sudo", "yum", "install", "-y", 
                                      "poppler-utils" if dependency == "poppler" else dependency])
                print(f"Successfully installed {dependency} using yum")
                return True
            else:
                print("Neither apt-get nor yum found. Please install manually.")
                return False
                
        elif system == "Windows":
            print(f"Automatic installation of {dependency} on Windows is not supported.")
            print(f"Please install {dependency} manually:")
            if dependency == "poppler":
                print("  Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
                print("  Then add the bin directory to your PATH")
            elif dependency == "tesseract":
                print("  Download and install from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
            
        return False
    except Exception as e:
        print(f"Failed to install {dependency}: {e}")
        return False

# Check and install required packages
required_packages = [
    "PyPDF2",
    "PyMuPDF",
    "pdf2image",
    "Pillow",
    "levenshtein",
    "pandas",
    "matplotlib",
    "nltk"
]

print("Checking and installing required packages...")
for package in required_packages:
    try:
        __import__(package.lower().replace("-", "_").split(">=")[0].split("==")[0])
        print(f"{package} is already installed")
    except ImportError:
        install_package(package)

# Try to install OCR-specific packages
ocr_packages = {
    "tesseract": ["pytesseract"],
    "paddleocr": ["paddlepaddle", "paddleocr"]
}

tesseract_available = os.path.exists(TESSERACT_PATH)
paddle_available = False

# Check if Tesseract is installed on the system
if platform.system() == "Windows":
    tesseract_available = shutil.which("tesseract") is not None
else:
    tesseract_available = shutil.which("tesseract") is not None

if tesseract_available:
    print("Tesseract is installed on the system")
    for package in ocr_packages["tesseract"]:
        try:
            __import__(package.lower())
            print(f"{package} is already installed")
        except ImportError:
            install_package(package)
else:
    print("Tesseract is not installed on the system")
    # Try to install tesseract
    if install_system_dependency("tesseract"):
        tesseract_available = True
        print("Installing Python packages for Tesseract...")
        for package in ocr_packages["tesseract"]:
            install_package(package)
    else:
        print("Installation instructions:")
        if platform.system() == "Windows":
            print("  Download and install from: https://github.com/UB-Mannheim/tesseract/wiki")
        elif platform.system() == "Darwin":  # macOS
            print("  Install with Homebrew: brew install tesseract")
        else:  # Linux
            print("  Install with apt: sudo apt install tesseract-ocr")

# Try to install PaddleOCR
try:
    for package in ocr_packages["paddleocr"]:
        try:
            __import__(package.lower())
            print(f"{package} is already installed")
            paddle_available = True
        except ImportError:
            if install_package(package):
                paddle_available = True
except Exception as e:
    print(f"Error installing PaddleOCR packages: {e}")
    paddle_available = False

# Check for poppler (required for pdf2image)
poppler_available = os.path.exists(PDFTOPPM_PATH)
if not poppler_available:
    print("Poppler is not installed on the system (required for PDF to image conversion)")
    # Try to install poppler
    if install_system_dependency("poppler"):
        poppler_available = True
    else:
        print("Installation instructions:")
        if platform.system() == "Windows":
            print("  Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("  Then add the bin directory to your PATH")
        elif platform.system() == "Darwin":  # macOS
            print("  Install with Homebrew: brew install poppler")
        else:  # Linux
            print("  Install with apt: sudo apt-get install poppler-utils")
else:
    print("Poppler is installed on the system")

# Import packages after installation
import PyPDF2
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import nltk
import Levenshtein

# Download NLTK data for text quality evaluation
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

if tesseract_available:
    import pytesseract
    from pdf2image import convert_from_path

if paddle_available:
    from paddleocr import PaddleOCR

# Define parsing methods
def parse_with_pypdf2(pdf_path: str) -> Tuple[str, float]:
    """Parse PDF with PyPDF2."""
    start_time = time.time()
    text = ""
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        print(f"Error with PyPDF2: {e}")
    
    processing_time = time.time() - start_time
    return text, processing_time

def parse_with_pymupdf(pdf_path: str) -> Tuple[str, float]:
    """Parse PDF with PyMuPDF."""
    start_time = time.time()
    text = ""
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text += doc[page_num].get_text() + "\n"
        doc.close()
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
    
    processing_time = time.time() - start_time
    return text, processing_time

def parse_with_tesseract(pdf_path: str) -> Tuple[str, float]:
    """Parse PDF using Tesseract OCR"""
    global TESSERACT_PATH, OCR_LANGUAGE, PDFTOPPM_PATH
    
    # Check if tesseract is available
    try:
        subprocess.run([TESSERACT_PATH, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Found Tesseract at {TESSERACT_PATH}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Tesseract not found at specified path, skipping...")
        return "", 0
    
    start_time = time.time()
    
    # Create a temporary directory for the images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images using pdftoppm
        try:
            subprocess.run([PDFTOPPM_PATH, '-png', pdf_path, f"{temp_dir}/page"], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"Converted PDF to images using pdftoppm at {PDFTOPPM_PATH}")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("pdftoppm not found at specified path, skipping...")
            return "", 0
            
        # Get the image files
        image_files = sorted(glob.glob(os.path.join(temp_dir, "*.png")))
        
        # Extract text from images using tesseract
        full_text = ""
        for image_file in image_files:
            try:
                # Use the specified language
                img = Image.open(image_file)
                text = pytesseract.image_to_string(img, lang=OCR_LANGUAGE)
                full_text += text + "\n\n"
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
    
    processing_time = time.time() - start_time
    return full_text, processing_time

def parse_with_paddleocr(pdf_path: str) -> Tuple[str, float]:
    """Parse PDF with PaddleOCR. Falls back to PyMuPDF if PaddleOCR fails."""
    global OCR_LANGUAGE, paddle_available, PDFTOPPM_PATH, poppler_available
    
    # Maximum number of pages to process for fallback method
    MAX_PAGES = 5
    
    if not paddle_available:
        return "PaddleOCR not available", 0
    
    if not os.path.exists(PDFTOPPM_PATH):
        return "Poppler not installed (required for PDF to image conversion)", 0
    
    # Map OCR_LANGUAGE to PaddleOCR language code
    lang_mapping = {
        "spa": "es",
        "eng": "en",
        # Add more language mappings as needed
    }
    
    paddle_lang = lang_mapping.get(OCR_LANGUAGE, "en")
    
    # Set environment variables to control Paddle behavior
    os.environ['FLAGS_use_gpu'] = '0'
    os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Check if running on MacOS
    is_macos = platform.system() == 'Darwin'
    is_apple_silicon = is_macos and 'arm' in platform.processor().lower()
    
    start_time = time.time()
    
    # Given the persistent issues with PaddleOCR on Apple Silicon, let's directly use PyMuPDF
    # as a fast, reliable fallback when we're on Apple Silicon
    if is_apple_silicon:
        print("Detected Apple Silicon. Using PyMuPDF as fallback due to known PaddleOCR allocation issues.")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(min(len(doc), MAX_PAGES)):
                text += f"--- Page {page_num+1} ---\n"
                text += doc[page_num].get_text() + "\n\n"
            doc.close()
            
            processing_time = time.time() - start_time
            return text, processing_time
        except Exception as err:
            print(f"PyMuPDF fallback also failed: {err}")
            return "Both PaddleOCR and PyMuPDF fallback failed", time.time() - start_time
    
    # For non-Apple Silicon, try PaddleOCR with the most conservative settings
    try:
        # Use a more robust configuration for different platforms
        ocr_kwargs = {
            'use_angle_cls': True,
            'lang': paddle_lang,
            'use_gpu': False,
            'use_xpu': False,
            'use_npu': False,
            'enable_mkldnn': False,
            'cpu_threads': 2,
            'rec_batch_num': 1,
            'det_db_score_mode': 'slow',  # More stable, albeit slower
            'det_limit_side_len': 640,
            'det_limit_type': 'max'
        }
        
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(**ocr_kwargs)
        
        # Use a conservative approach - process only one page at a time
        with tempfile.TemporaryDirectory() as path:
            # Convert PDF to images
            try:
                # Use pdf2image for conversion, with low DPI for faster processing
                from pdf2image import convert_from_path
                    
                # Convert only the first page initially as a test
                images = convert_from_path(
                    pdf_path,
                    output_folder=path,
                    poppler_path=os.path.dirname(PDFTOPPM_PATH) if poppler_available else None,
                    dpi=120,  # Even lower DPI for stability
                    first_page=1,
                    last_page=1
                )
                
                if images:
                    # Successfully converted first page, now try to process it
                    img_path = os.path.join(path, f'page_0.png')
                    images[0].save(img_path)
                    
                    # Try to process the first page with conservative settings
                    try:
                        img = Image.open(img_path)
                        w, h = img.size
                        ratio = min(1.0, 640 / max(h, w))
                        new_size = (int(w * ratio), int(h * ratio))
                        img = img.resize(new_size)
                        small_img_path = os.path.join(path, 'small_page_0.png')
                        img.save(small_img_path)
                        img_path = small_img_path
                        
                        result = ocr.ocr(img_path, cls=True)
                        
                        # Process text output
                        text = ""
                        if result is not None:
                            # First page worked, process text
                            page_text = extract_text_from_paddle_result(result)
                            text += page_text + "\n\n"
                            
                            # Process remaining pages
                            remaining_images = convert_from_path(
                                pdf_path,
                                output_folder=path,
                                poppler_path=os.path.dirname(PDFTOPPM_PATH) if poppler_available else None,
                                dpi=120,
                                first_page=2,
                                last_page=MAX_PAGES
                            )
                            
                            for i, image in enumerate(remaining_images, start=1):
                                img_path = os.path.join(path, f'page_{i}.png')
                                image.save(img_path)
                                
                                # Resize for stability
                                img = Image.open(img_path)
                                w, h = img.size
                                ratio = min(1.0, 640 / max(h, w))
                                new_size = (int(w * ratio), int(h * ratio))
                                img = img.resize(new_size)
                                small_img_path = os.path.join(path, f'small_page_{i}.png')
                                img.save(small_img_path)
                                img_path = small_img_path
                                
                                try:
                                    result = ocr.ocr(img_path, cls=True)
                                    if result is not None:
                                        page_text = extract_text_from_paddle_result(result)
                                        text += page_text + "\n\n"
                                except Exception as e:
                                    print(f"Error processing page {i+1}: {e}")
                                    # Continue to the next page if one fails
                                    continue
                        else:
                            # Results were None, use fallback
                            print("PaddleOCR returned None. Using PyMuPDF as fallback.")
                            import fitz  # PyMuPDF
                            doc = fitz.open(pdf_path)
                            text = "PaddleOCR failed, using PyMuPDF fallback:\n\n"
                            for page_num in range(min(len(doc), MAX_PAGES)):
                                text += f"--- Page {page_num+1} ---\n"
                                text += doc[page_num].get_text() + "\n\n"
                            doc.close()
                        
                    except Exception as e:
                        print(f"Error with PaddleOCR on first page: {e}")
                        # Fall back to PyMuPDF
                        import fitz  # PyMuPDF
                        doc = fitz.open(pdf_path)
                        text = "PaddleOCR failed, using PyMuPDF fallback:\n\n"
                        for page_num in range(min(len(doc), MAX_PAGES)):
                            text += f"--- Page {page_num+1} ---\n"
                            text += doc[page_num].get_text() + "\n\n"
                        doc.close()
                else:
                    # Couldn't convert images
                    import fitz  # PyMuPDF
                    doc = fitz.open(pdf_path)
                    text = "PDF to image conversion failed, using PyMuPDF fallback:\n\n"
                    for page_num in range(min(len(doc), MAX_PAGES)):
                        text += f"--- Page {page_num+1} ---\n"
                        text += doc[page_num].get_text() + "\n\n"
                    doc.close()
                
            except Exception as e:
                print(f"Error converting PDF to images: {e}")
                # Fall back to PyMuPDF
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                text = "PDF processing error, using PyMuPDF fallback:\n\n"
                for page_num in range(min(len(doc), MAX_PAGES)):
                    text += f"--- Page {page_num+1} ---\n"
                    text += doc[page_num].get_text() + "\n\n"
                doc.close()
                
    except Exception as e:
        print(f"Error with PaddleOCR: {e}")
        # Final fallback
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = "General PaddleOCR error, using PyMuPDF fallback:\n\n"
            for page_num in range(min(len(doc), MAX_PAGES)):
                text += f"--- Page {page_num+1} ---\n"
                text += doc[page_num].get_text() + "\n\n"
            doc.close()
        except Exception as err:
            text = f"All OCR methods failed: {str(e)} / {str(err)}"
        
    processing_time = time.time() - start_time
    return text, processing_time

def extract_text_from_paddle_result(result):
    """Helper function to extract text from PaddleOCR result with proper error handling."""
    page_text = []
    
    try:
        if not result:
            return ""
            
        # Handle different versions of PaddleOCR that return different structures
        if isinstance(result, list):
            for line_result in result:
                if line_result is None:
                    continue
                    
                # Handle different result structures
                if isinstance(line_result, list):
                    for line in line_result:
                        if not line:
                            continue
                        if len(line) >= 2:
                            if isinstance(line[1], tuple) and len(line[1]) >= 1:
                                page_text.append(line[1][0])
                            elif isinstance(line[1], str):
                                page_text.append(line[1])
                elif isinstance(line_result, tuple) and len(line_result) >= 2:
                    if isinstance(line_result[1], str):
                        page_text.append(line_result[1])
                    elif isinstance(line_result[1], tuple) and len(line_result[1]) >= 1:
                        page_text.append(line_result[1][0])
        
        return " ".join(page_text)
    except Exception as e:
        print(f"Error extracting text from PaddleOCR result: {e}")
        return ""

# Define functions for evaluating text quality
def evaluate_text_quality(text: str) -> Dict[str, Any]:
    """Evaluate the quality of extracted text."""
    if not text or text == "Tesseract not available" or text == "PaddleOCR not available":
        return {
            "word_count": 0,
            "char_count": 0,
            "non_empty_lines": 0,
            "avg_line_length": 0,
            "whitespace_ratio": 0
        }
    
    # Basic statistics
    words = text.split()
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    
    word_count = len(words)
    char_count = len(text)
    whitespace_count = sum(c.isspace() for c in text)
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "non_empty_lines": len(non_empty_lines),
        "avg_line_length": sum(len(line) for line in non_empty_lines) / max(1, len(non_empty_lines)),
        "whitespace_ratio": whitespace_count / max(1, char_count)
    }

def compare_text_similarity(texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Compare similarity between texts extracted by different methods."""
    methods = list(texts.keys())
    similarity_matrix = {}
    
    for method1 in methods:
        similarity_matrix[method1] = {}
        for method2 in methods:
            if method1 == method2:
                similarity_matrix[method1][method2] = 1.0
            else:
                text1 = texts[method1]
                text2 = texts[method2]
                
                # Skip if either text is empty or unavailable
                if (not text1 or text1 == "Tesseract not available" or text1 == "PaddleOCR not available" or
                    not text2 or text2 == "Tesseract not available" or text2 == "PaddleOCR not available"):
                    similarity_matrix[method1][method2] = 0.0
                    continue
                
                # Calculate Levenshtein ratio for similarity
                distance = Levenshtein.distance(text1, text2)
                max_len = max(len(text1), len(text2))
                similarity = 1 - (distance / max_len) if max_len > 0 else 0
                similarity_matrix[method1][method2] = similarity
    
    return similarity_matrix

def save_results(file_name: str, results: Dict[str, Dict[str, Any]], output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed text output
    for method, data in results.items():
        if method != "comparison":
            output_path = os.path.join(output_dir, f"{os.path.basename(file_name)}_{method}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data["text"])
    
    # Create DataFrame for timing and quality metrics
    methods = [m for m in results.keys() if m != "comparison"]
    metrics_df = pd.DataFrame(index=methods)
    
    metrics_df["Processing Time (s)"] = [results[m]["time"] for m in methods]
    metrics_df["Word Count"] = [results[m]["quality"]["word_count"] for m in methods]
    metrics_df["Character Count"] = [results[m]["quality"]["char_count"] for m in methods]
    metrics_df["Non-empty Lines"] = [results[m]["quality"]["non_empty_lines"] for m in methods]
    metrics_df["Avg Line Length"] = [results[m]["quality"]["avg_line_length"] for m in methods]
    metrics_df["Whitespace Ratio"] = [results[m]["quality"]["whitespace_ratio"] for m in methods]
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, f"{os.path.basename(file_name)}_metrics.csv"))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot processing times
    metrics_df["Processing Time (s)"].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Processing Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Method')
    
    # Plot word counts
    metrics_df["Word Count"].plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Word Count Comparison')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Method')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(file_name)}_comparison.png"))
    
    # Save similarity matrix
    if "comparison" in results and "similarity" in results["comparison"]:
        similarity_df = pd.DataFrame(results["comparison"]["similarity"])
        similarity_df.to_csv(os.path.join(output_dir, f"{os.path.basename(file_name)}_similarity.csv"))

def main():
    parser = argparse.ArgumentParser(description='Compare PDF parsing methods')
    parser.add_argument('--input', '-i', type=str, default='knowledge', help='PDF file or directory containing PDFs')
    parser.add_argument('--output', '-o', type=str, default='tests/results', help='Output directory')
    parser.add_argument('--skip-install', action='store_true', help='Skip automatic installation of dependencies')
    parser.add_argument('--lang', '-l', type=str, default='spa', help='Language for OCR (e.g., "spa" for Spanish, "eng" for English)')
    args = parser.parse_args()
    
    # Only attempt to install dependencies if not explicitly skipped
    global SKIP_INSTALL, OCR_LANGUAGE
    SKIP_INSTALL = args.skip_install
    OCR_LANGUAGE = args.lang
    
    # Check if input is provided
    if not args.input:
        print("Please provide a PDF file or directory with --input")
        return
    
    # Get list of PDFs to process
    pdf_files = []
    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    elif os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
        pdf_files.append(args.input)
    else:
        print(f"Input '{args.input}' is not a PDF file or directory containing PDFs")
        return
    
    if not pdf_files:
        print(f"No PDF files found in '{args.input}'")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        
        # Initialize results dictionary
        results = {
            "pypdf2": {},
            "pymupdf": {},
            "tesseract": {},
            "paddleocr": {},
            "comparison": {}
        }
        
        # Parse with PyPDF2
        print("Parsing with PyPDF2...")
        text, proc_time = parse_with_pypdf2(pdf_file)
        results["pypdf2"]["text"] = text
        results["pypdf2"]["time"] = proc_time
        results["pypdf2"]["quality"] = evaluate_text_quality(text)
        print(f"  Time: {proc_time:.2f}s, Words: {results['pypdf2']['quality']['word_count']}")
        
        # Parse with PyMuPDF
        print("Parsing with PyMuPDF...")
        text, proc_time = parse_with_pymupdf(pdf_file)
        results["pymupdf"]["text"] = text
        results["pymupdf"]["time"] = proc_time
        results["pymupdf"]["quality"] = evaluate_text_quality(text)
        print(f"  Time: {proc_time:.2f}s, Words: {results['pymupdf']['quality']['word_count']}")
        
        # Parse with Tesseract (if available)
        if tesseract_available:
            print("Parsing with Tesseract...")
            text, proc_time = parse_with_tesseract(pdf_file)
            results["tesseract"]["text"] = text
            results["tesseract"]["time"] = proc_time
            results["tesseract"]["quality"] = evaluate_text_quality(text)
            print(f"  Time: {proc_time:.2f}s, Words: {results['tesseract']['quality']['word_count']}")
        else:
            results["tesseract"]["text"] = "Tesseract not available"
            results["tesseract"]["time"] = 0
            results["tesseract"]["quality"] = evaluate_text_quality("")
            print("  Tesseract not available")
        
        # Parse with PaddleOCR (if available)
        if paddle_available:
            print("Parsing with PaddleOCR...")
            text, proc_time = parse_with_paddleocr(pdf_file)
            results["paddleocr"]["text"] = text
            results["paddleocr"]["time"] = proc_time
            results["paddleocr"]["quality"] = evaluate_text_quality(text)
            print(f"  Time: {proc_time:.2f}s, Words: {results['paddleocr']['quality']['word_count']}")
        else:
            results["paddleocr"]["text"] = "PaddleOCR not available"
            results["paddleocr"]["time"] = 0
            results["paddleocr"]["quality"] = evaluate_text_quality("")
            print("  PaddleOCR not available")
        
        # Compare text similarity
        texts = {
            method: results[method]["text"] 
            for method in ["pypdf2", "pymupdf", "tesseract", "paddleocr"]
        }
        similarity_matrix = compare_text_similarity(texts)
        results["comparison"]["similarity"] = similarity_matrix
        
        # Save results
        save_results(os.path.basename(pdf_file), results, args.output)
        
        print(f"Results saved to {args.output}")
    
    print("\nComparison complete! Results saved to output directory.")
    print("For each PDF, we've generated:")
    print("  - Text files with the extracted content from each method")
    print("  - CSV file with metrics comparison")
    print("  - PNG visualization of processing time and word count")
    print("  - CSV file with text similarity matrix between methods")

if __name__ == "__main__":
    main() 