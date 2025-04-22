# ZChatBot - An Industrial RAG Chatbot

ZChatBot is a Retrieval-Augmented Generation (RAG) chatbot designed for industrial use cases. It uses local LLM models to provide private, secure responses based on your own documents.

## Features

- 💬 Question answering about industrial equipment and processes
- 📄 Document processing (PDF, DOCX, TXT, images) with advanced OCR
- 🔍 Retrieval Augmented Generation (RAG) for accurate, context-aware responses
- 🚀 Multiple LLM backend options (local models and OpenAI API)
- 🖥️ Simple web interface for easy interaction

## Quick Setup

To set up the chatbot, follow these simple steps:

1. Clone the repository:
```
git clone <repository-url>
cd ZChatBot
```

2. (Recommended) Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**MacOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your command prompt, indicating the virtual environment is active.

3. Run the setup script:
```
python setup.py
```

This will:
- Install all required dependencies
- Download the appropriate model (Phi-3-mini-4k-instruct)
- Create necessary directories for your documents and data

4. Verify installation:
```
# Check if key packages are installed correctly
python -c "import torch; import gradio; print('Setup successful!')"
```

5. Add documents to the knowledge base:

```bash
# Put your PDFs, DOCXs, TXTs in the knowledge directory
mkdir -p knowledge
# Copy your files to the knowledge directory
```

6. Run the web interface:

```bash
python webapp.py --process
```

7. Access the web interface at http://localhost:7860

## Usage

After setup, you can use the chatbot with two simple commands:

1. Process your documents:
```
python main.py --process
```
This command will process any documents you place in the `knowledge` directory.

2. Start the interactive chat mode:
```
python main.py --interactive
```
This will start the chatbot in interactive mode where you can ask questions about your documents.

## Web Interface

ZChatBot comes with a web interface powered by Gradio, making it easy to interact with your chatbot through a browser.

1. Install Gradio (if you haven't already run setup.py):
```
pip install gradio
```

2. Launch the web interface:
```
python webapp.py
```

3. Open your browser and navigate to:
```
http://localhost:7860
```

The web interface provides:
- A chat interface for asking questions
- A model selector to switch between different LLMs
- Tools to process and manage your knowledge base

To make the interface accessible from other computers on your network, use:
```
python webapp.py -- --server_name="0.0.0.0"
```

To create a temporary public URL (for sharing), use:
```
python webapp.py -- --share=True
```

## Supported Models

The chatbot includes configurations for:
- Phi-3-mini-4k-instruct (default) - A smaller but powerful model with 4K context window
- TinyLlama-1.1B - A lightweight alternative for minimal hardware (~600MB)
- Mistral-7B-instruct-16k - A larger model with extended 16K context window
- OpenAI models (GPT-3.5-Turbo, GPT-4) - API-based models (requires API key)

To switch models, type `switch:<model_name>` in interactive mode.

## System Requirements

- Python 3.8+
- 8GB RAM minimum (16GB+ recommended for larger models)
- 4GB of free disk space for models
- GPU acceleration supported but not required

## Troubleshooting

If you encounter issues with model loading:

1. Make sure the model file was downloaded correctly during setup
2. For Mac users, the setup should automatically configure Metal support for GPU acceleration
3. On Windows, ensure you have the latest Python and pip versions

For other issues, please check the error messages in the console output.

## Available Models

The system supports both local models and the OpenAI API:

### Local Models
- **tinyllama-1.1b** - Very small model for limited hardware (600MB)
- **phi-3-mini-4k-instruct** - Small model with good performance (2.2GB)
- **mistral-7b-instruct-16k** - Extended 16K context window for large technical documents (4.4GB)

### OpenAI API Models
- **gpt-3.5-turbo** - Affordable API model with 16k context window (requires API key)
- **gpt-4** - Premium API model with advanced reasoning (more expensive, requires API key)

## Using OpenAI Models

To use OpenAI models:

1. Get an API key from [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Add your API key to `.env` file: `OPENAI_API_KEY="your-key-here"`
3. Run with the OpenAI model: `python webapp.py --model gpt-3.5-turbo`

## Document Processing

The system can process:
- PDF documents (both text and scanned using OCR)
- Word documents (DOCX)
- Text files (TXT)
- Images (PNG, JPG, JPEG) via OCR

OCR is handled through PaddleOCR with PyMuPDF for PDF processing, providing high-quality text extraction.

For multilingual document support, the system uses the "intfloat/multilingual-e5-large" embedding model, which provides excellent cross-lingual understanding for document retrieval.

## Technology Evaluations

The project includes an `tools_evaluations` folder (previously named "tools_evaluations") containing comparison scripts for various technologies:

- `pdf_parsing_comparison.py` - Benchmarks different PDF parsing libraries
- Other technology evaluations for document processing

These scripts help evaluate the performance and accuracy of different libraries and approaches for document processing tasks. They are not unit tests but rather benchmark tools for comparing technologies.

To run a comparison:
```bash
python tools_evaluations/pdf_parsing_comparison.py
```

## License

[MIT License](LICENSE) 