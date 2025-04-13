# Industrial RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed for industrial environments, powered by Mistral 7B and running entirely on-premises.

## Features

- Processes and indexes various document formats (PDF, DOCX, TXT, images)
- Uses a vector database to store document embeddings
- Runs a state-of-the-art open-source LLM (Mistral 7B) locally
- Answers questions based on your custom knowledge base
- Completely on-premises - no cloud APIs required

## Project Structure

```
industrial_chatbot/
├── data/                   # Vector store data
├── knowledge/              # Knowledge base documents
├── models/                 # LLM model files
├── src/                    # Source code
│   ├── __init__.py
│   ├── chatbot.py          # Main chatbot class
│   ├── config.py           # Configuration settings
│   ├── document_processor.py  # Handles different document types
│   ├── llm.py              # Interface to local LLM
│   └── vector_store.py     # Vector DB interface
├── main.py                 # CLI interface
├── README.md               # This file
└── requirements.txt        # Dependencies
```

## Setup

1. Create a virtual environment and activate it:

```bash
# Navigate to the project directory
cd industrial_chatbot

# Activate the virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR (for image processing):

```bash
# For macOS
brew install tesseract

# For Ubuntu/Debian
sudo apt-get install tesseract-ocr

# For Windows
# Download and install from https://github.com/UB-Mannheim/tesseract/wiki
```

4. Download the GGUF model:

```bash
mkdir -p models/mistral-7b-instruct-v0.2-gguf
curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -o models/mistral-7b-instruct-v0.2-gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## GPU Acceleration

The chatbot is configured to use GPU acceleration for faster responses. To enable GPU support:

1. Make sure you have a compatible NVIDIA GPU

2. Install CUDA toolkit (version 11.7+ recommended):
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your operating system

3. Install PyTorch with CUDA support:
   ```bash
   # For CUDA 11.8
   pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
   ```

4. Verify GPU is detected:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   ```

The chatbot's `gpu_layers` parameter in `src/llm.py` is set to 32 by default, which offloads a significant portion of the model to GPU. Adjust based on your GPU memory:
- For GPUs with 8GB+ VRAM: Use 32 or higher
- For GPUs with less VRAM: Try a lower number (e.g., 24, 16, or 8)
- For CPU-only: Set to 0

## Adding Knowledge

Place your documents in the `knowledge` directory:

```bash
# Create the knowledge directory if it doesn't exist
mkdir -p knowledge

# Copy your files
cp /path/to/your/documents/*.pdf knowledge/
cp /path/to/your/documents/*.docx knowledge/
# etc.
```

## Usage

### Process Documents

Before asking questions, you need to process your documents and build the vector database:

```bash
python main.py --process
```

### Interactive Mode

Start an interactive session with the chatbot:

```bash
python main.py --interactive
```

### Single Query

Ask a single question:

```bash
python main.py --query "What is the operating temperature of pump XYZ-123?"
```

### Clear Knowledge Base

If you need to rebuild your knowledge base from scratch:

```bash
python main.py --clear
```

## Upgrading to Mixtral 8x7B

For production use, you can upgrade to Mixtral 8x7B by modifying the configuration in `src/config.py`. Change:

```python
MODEL_ID = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
MODEL_TYPE = "mixtral"
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "mixtral-8x7b-instruct-v0.1-gguf")
```

Then download the new model:

```bash
mkdir -p models/mixtral-8x7b-instruct-v0.1-gguf
curl -L https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf -o models/mixtral-8x7b-instruct-v0.1-gguf/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
```

## Troubleshooting

If you encounter issues with loading the LLM:

1. Make sure you've downloaded the correct GGUF file format of the model
2. Check that your system meets the minimum requirements (8GB RAM for Mistral 7B with Q4_K_M quantization)
3. For larger models, consider using a machine with more RAM or a GPU

For GPU-related issues:
- Verify CUDA installation with `nvidia-smi` command
- Check PyTorch can see your GPU with `torch.cuda.is_available()`
- Try reducing `gpu_layers` if you encounter CUDA out of memory errors
- Update graphics drivers to the latest version

## License

This project is open source and available under the MIT License. 