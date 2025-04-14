import os
from pathlib import Path
import platform
import sys
import torch

# Project directories
BASE_DIR = Path(__file__).parent.parent.absolute()
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Vector DB settings
VECTORDB_PATH = os.path.join(DATA_DIR, "vectordb")

# Detect if using CUDA
is_cuda = torch.cuda.is_available() if 'torch' in sys.modules else False

# Map model types differently based on hardware
PHI3_MODEL_TYPE = "phi" if is_cuda else "phi"

# Available models configuration
AVAILABLE_MODELS = {
    # "mistral-7b": {
    #     "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    #     "model_type": "mistral",
    #     "local_path": os.path.join(MODELS_DIR, "mistral-7b-instruct-v0.2-gguf"),
    #     "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    #     "download_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    # },
    "zephyr-7b": {
        "model_id": "TheBloke/zephyr-7B-beta-GGUF",
        "model_type": "mistral",
        "local_path": os.path.join(MODELS_DIR, "zephyr-7b-beta-gguf"),
        "filename": "zephyr-7b-beta.Q4_K_M.gguf",
        "download_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
    },
#     "phi-3-mini": {
#        "model_id": "bartowski/Phi-3-mini-4k-instruct-GGUF",
#        "model_type": "phi",
#        "local_path": os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct-gguf"),
#        "filename": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
#        "download_url": "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
#    },
    "phi-3-mini-cuda": {
       "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
       "model_type": "phi",
       "local_path": os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct-gguf"),
       "filename": "Phi-3-mini-4k-instruct-q4.gguf",
       "download_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
   },
    # "phi-3-medium": {
    #     "model_id": "bartowski/Phi-3-medium-4k-instruct-GGUF",
    #     "model_type": PHI3_MODEL_TYPE,
    #     "local_path": os.path.join(MODELS_DIR, "phi-3-medium-4k-instruct-gguf"),
    #     "filename": "Phi-3-medium-4k-instruct-Q4_K_M.gguf",
    #     "download_url": "https://huggingface.co/bartowski/Phi-3-medium-4k-instruct-GGUF/resolve/main/Phi-3-medium-4k-instruct-Q4_K_M.gguf"
    # },
    "tinyllama-1.1b": {
        "model_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "model_type": "llama",
        "local_path": os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0-gguf"),
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "download_url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    },
    "mixtral-8x7b": {
        "model_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "model_type": "mistral",
        "local_path": os.path.join(MODELS_DIR, "mixtral-8x7b-instruct-v0.1-gguf"),
        "filename": "mixtral-8x7b-instruct-v0.1.Q2_K.gguf",
        "download_url": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q2_K.gguf",
        "fallback_filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "fallback_path": os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0-gguf")
    },
}

# Default model
DEFAULT_MODEL = "phi-3-mini-cuda"
MODEL_CONFIG = AVAILABLE_MODELS[DEFAULT_MODEL]

# Get specific model settings from the active model config
MODEL_ID = MODEL_CONFIG["model_id"]
MODEL_TYPE = MODEL_CONFIG["model_type"]
LOCAL_MODEL_PATH = MODEL_CONFIG["local_path"]
MODEL_FILENAME = MODEL_CONFIG["filename"]

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 1024

# LLM inference settings
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_LENGTH = 4096
LLM_GPU_LAYERS = 32  # Number of layers to offload to GPU when acceleration is available
LLM_VERBOSE = False

# Model-specific configurations for different backends
LLAMACPP_CONFIGS = {
    # Default configuration for all models using LlamaCpp
    "default": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": LLM_CONTEXT_LENGTH,
        "n_batch": 512,
        "n_gpu_layers": LLM_GPU_LAYERS,
        "f16_kv": True,
        "verbose": LLM_VERBOSE,
        "seed": 42
    },
    # Mixtral-specific configuration (optimized for memory efficiency)
    "mixtral-8x7b": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": 2048,               # Reduced context to save memory
        "n_batch": 256,              # Smaller batch size for memory efficiency
        "n_gpu_layers": 1,           # Limit GPU usage
        "f16_kv": True,              # Half precision for key/value cache
        "use_mlock": False,          # Don't lock memory
        "offload_kqv": True,         # Offload key/query/value tensors
        "verbose": True,
        "seed": 42                   # For reproducibility
    },
    # Phi-specific configuration
    "phi-3-mini": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": LLM_CONTEXT_LENGTH,
        "n_batch": 512,
        "n_gpu_layers": LLM_GPU_LAYERS,
        "f16_kv": True,
        "verbose": LLM_VERBOSE,
        "seed": 42
    },
    "phi-3-mini-cuda": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": LLM_CONTEXT_LENGTH,
        "n_batch": 512,
        "n_gpu_layers": LLM_GPU_LAYERS,
        "f16_kv": True,
        "verbose": LLM_VERBOSE,
        "seed": 42
    },
    # Phi-3-medium configuration
    "phi-3-medium": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": 2048,               # Reduced context to save memory
        "n_batch": 128,              # Smaller batch size for memory efficiency
        "n_gpu_layers": 16,          # Reduced from 32 to balance performance and memory usage
        "f16_kv": True,              # Half precision for key/value cache
        "use_mlock": False,          # Don't lock memory
        "offload_kqv": True,         # Offload key/query/value tensors
        "verbose": LLM_VERBOSE,
        "seed": 42
    },
    # Fallback configuration for tiny models (CPU only)
    "fallback": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": 2048,
        "n_batch": 256,
        "n_gpu_layers": 0,           # CPU only
        "verbose": True
    }
}

# CTransformers configurations for different models
CTRANSFORMERS_CONFIGS = {
    "default": {
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': LLM_TEMPERATURE,
        'context_length': LLM_CONTEXT_LENGTH,
        'gpu_layers': LLM_GPU_LAYERS
    }
}

# Model-specific stop sequences for generation
STOP_SEQUENCES = {
    "phi": ["<|user|>", "Question:", "Context:"],
    "mistral": []  # Add any stop sequences for mistral models if needed
}

# Prompt templates for different model types
PROMPT_TEMPLATES = {
    "llama": """<|system|>
You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}</s>
<|user|>
{query}</s>
<|assistant|>""",
    
    "phi": """<|system|>
You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
<|user|>
Context:
{context}

Question: {query}
<|assistant|>""",
    
    "mistral": """[INST] You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query} [/INST]"""
}

# Create directories if they don't exist
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VECTORDB_PATH, exist_ok=True)

# Create model directories
for model_name, config in AVAILABLE_MODELS.items():
    os.makedirs(config["local_path"], exist_ok=True) 