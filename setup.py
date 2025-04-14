#!/usr/bin/env python3
import os
import subprocess
import sys
import platform

def run_command(command, cwd=None):
    """Run a shell command and print output."""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, 
                              capture_output=True, cwd=cwd)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    # Get the current directory
    current_dir = os.getcwd()
    
    print("Setting up ZChatBot...")
    
    # Create necessary directories
    for directory in ["models", "data", "knowledge"]:
        dir_path = os.path.join(current_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    required_packages = [
        "torch",
        "langchain",
        "langchain_community",
        "ctransformers",
        "llama-cpp-python",
        "transformers",
        "accelerate",
        "sentence-transformers",
        "pypdf",
        "docx2txt",
        "Pillow",
        "pytesseract",
        "unstructured",
        "pdf2image",
        "faiss-cpu",
        "chromadb",
        "huggingface_hub",
        "einops",
        "tqdm",
        "pydantic",
        "python-dotenv"
    ]
    
    # Check for Apple Silicon to install optimized llama-cpp-python
    is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
    
    # Basic packages first
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install packages in chunks to avoid long command lines
    chunk_size = 5
    for i in range(0, len(required_packages), chunk_size):
        chunk = required_packages[i:i+chunk_size]
        run_command(f"{sys.executable} -m pip install {' '.join(chunk)}")
    
    # Install Gradio explicitly
    print("\nInstalling Gradio web interface...")
    run_command(f"{sys.executable} -m pip install gradio --no-cache-dir")
    
    # Special handling for llama-cpp-python on Apple Silicon
    if is_apple_silicon:
        print("\nDetected Apple Silicon Mac, installing optimized llama-cpp-python with Metal support...")
        run_command(f"{sys.executable} -m pip uninstall llama-cpp-python --yes")
        run_command(f"CMAKE_ARGS=\"-DLLAMA_METAL=on\" {sys.executable} -m pip install --force-reinstall llama-cpp-python")
    
    # Phi-3 model directory
    phi3_dir = os.path.join(current_dir, "models", "phi-3-mini-4k-instruct-gguf")
    os.makedirs(phi3_dir, exist_ok=True)
    
    # Path to default model
    phi3_model_path = os.path.join(phi3_dir, "Phi-3-mini-4k-instruct-Q4_K_M.gguf")
    
    # Check if model exists, download if not
    if not os.path.exists(phi3_model_path):
        print(f"\nDownloading Phi-3 mini model to {phi3_model_path}...")
        download_url = "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
        
        # Try using curl first
        if platform.system() != "Windows" and run_command("curl --version"):
            run_command(f"curl -L {download_url} -o {phi3_model_path}")
        else:
            try:
                # Use Python requests as fallback
                import requests
                print("Starting model download (this may take a while)...")
                with requests.get(download_url, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    downloaded = 0
                    with open(phi3_model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192*1024):
                            if chunk:
                                downloaded += len(chunk)
                                f.write(chunk)
                                if total_size > 0:
                                    percent = (downloaded / total_size) * 100
                                    print(f"Download progress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end='\r')
                print("\nDownload complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("\nPlease download the model manually:")
                print(f"1. Download from: {download_url}")
                print(f"2. Save to: {phi3_model_path}")
    else:
        print(f"\nModel already exists at {phi3_model_path}")
    
    print("\n=== Setup Complete! ===")
    print("\nTo use the chatbot:")
    print("1. Process documents: python main.py --process")
    print("2. Start interactive mode: python main.py --interactive")

if __name__ == "__main__":
    main() 