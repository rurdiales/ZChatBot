# ZChatBot - An Industrial RAG Chatbot

ZChatBot is a Retrieval-Augmented Generation (RAG) chatbot designed for industrial use cases. It uses local LLM models to provide private, secure responses based on your own documents.

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
- Download the appropriate model (Phi-3-mini)
- Create necessary directories for your documents and data

4. Verify installation:
```
# Check if key packages are installed correctly
python -c "import torch; import gradio; print('Setup successful!')"
```

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
- Phi-3-mini (default) - A smaller but powerful model
- Phi-3-mini-128k - Extended context model for handling more text
- TinyLlama - A lightweight alternative for minimal hardware
- Zephyr-7B - A larger model for more complex reasoning

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