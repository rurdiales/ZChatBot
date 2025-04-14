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
VECTORDB_PATH = os.path.join(DATA_DIR, "vectordb")

# Hardware detection
IS_CUDA = torch.cuda.is_available() if 'torch' in sys.modules else False
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# LLM inference settings - base config for all models
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 1024
LLM_VERBOSE = False

# GPU settings
# Default to 16 layers for most models (balanced performance/memory)
# Use 0 for CPU-only mode
LLM_GPU_LAYERS = 16 if (IS_CUDA or IS_APPLE_SILICON) else 0

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model families
MODEL_FAMILIES = {
    "phi": {
        "model_type": "phi",
        "stop_words": ["<|user|>", "Question:", "Context:"],
        "prompt_template": """<|system|>
You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
<|user|>
Context:
{context}

Question: {query}
<|assistant|>"""
    },
    "mistral": {
        "model_type": "mistral",
        "stop_words": [],
        "prompt_template": """[INST] You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query} [/INST]"""
    },
    "llama": {
        "model_type": "llama",
        "stop_words": [],
        "prompt_template": """<|system|>
You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}</s>
<|user|>
{query}</s>
<|assistant|>"""
    },
    "qwen": {
        "model_type": "gpt2" if IS_APPLE_SILICON else "llama",
        "stop_words": [],
        "prompt_template": """<|im_start|>system
You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
    }
}

# Base LlamaCpp configuration template
LLAMACPP_BASE_CONFIG = {
    "temperature": LLM_TEMPERATURE,
    "max_tokens": MAX_NEW_TOKENS,
    "n_ctx": LLM_CONTEXT_LENGTH,
    "n_batch": 512,
    "n_gpu_layers": LLM_GPU_LAYERS,
    "f16_kv": True,
    "verbose": LLM_VERBOSE,
    "seed": 42
}

# Base optimized configuration template for memory-efficient models
LLAMACPP_OPTIMIZED_CONFIG = {
    "temperature": LLM_TEMPERATURE,
    "max_tokens": MAX_NEW_TOKENS,
    "n_ctx": 2048,              # Reduced context to save memory
    "n_batch": 256,             # Smaller batch size for efficiency
    "n_gpu_layers": LLM_GPU_LAYERS,
    "f16_kv": True,             # Half precision for key/value cache
    "use_mlock": False,         # Don't lock memory
    "verbose": LLM_VERBOSE,
    "seed": 42
}

# Available models configuration
AVAILABLE_MODELS = {
    # Phi-3 models
    # "phi-3-mini": {
    #     "model_id": "bartowski/Phi-3-mini-4k-instruct-GGUF",
    #     "family": "phi",
    #     "local_path": os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct-gguf"),
    #     "filename": "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    #     "download_url": "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    #     "config_template": "optimized",
    #     "description": "Small model with good performance (2.2GB)"
    # },
    "phi-3-medium": {
        "model_id": "bartowski/Phi-3-medium-4k-instruct-GGUF",
        "family": "phi",
        "local_path": os.path.join(MODELS_DIR, "phi-3-medium-4k-instruct-gguf"),
        "filename": "Phi-3-medium-4k-instruct-Q4_K_M.gguf",
        "download_url": "https://huggingface.co/bartowski/Phi-3-medium-4k-instruct-GGUF/resolve/main/Phi-3-medium-4k-instruct-Q4_K_M.gguf",
        "config_template": "optimized",
        "config_override": {
            "n_batch": 128,
        },
        "description": "Medium-sized model with better reasoning (4GB)"
    },
    "phi-3-mini-cuda": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "family": "phi",
        "local_path": os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct-gguf"),
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "download_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "config_template": "optimized",
        "description": "Small model with good performance (2.2GB)"
    },
    
    # Mistral models
    # "mistral-7b": {
    #     "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    #     "family": "mistral",
    #     "local_path": os.path.join(MODELS_DIR, "mistral-7b-instruct-v0.2-gguf"),
    #     "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    #     "download_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    #     "config_template": "optimized",
    #     "description": "Strong general-purpose model (4.1GB)"
    # },
    "mistral-7b-instruct-16k": {
        "model_id": "TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF",
        "family": "mistral",
        "local_path": os.path.join(MODELS_DIR, "openhermes-2.5-mistral-7b-16k-gguf"),
        "filename": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
        "download_url": "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-16k-GGUF/resolve/main/openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
        "config_template": "optimized",
        "config_override": {
            "n_ctx": 16384,          # Very large context window for extensive documents
            "n_batch": 128,          # Balanced batch size
            "n_gpu_layers": 16       # Optimized for most hardware
        },
        "description": "Extended 16K context window for large technical documents (4.4GB)"
    },
    "zephyr-7b": {
        "model_id": "TheBloke/zephyr-7B-beta-GGUF",
        "family": "mistral",
        "local_path": os.path.join(MODELS_DIR, "zephyr-7b-beta-gguf"),
        "filename": "zephyr-7b-beta.Q4_K_M.gguf",
        "download_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
        "config_template": "optimized",
        "description": "Instruction-tuned 7B model (4.1GB)"
    },
    # "mixtral-8x7b": {
    #     "model_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    #     "family": "mistral",
    #     "local_path": os.path.join(MODELS_DIR, "mixtral-8x7b-instruct-v0.1-gguf"),
    #     "filename": "mixtral-8x7b-instruct-v0.1.Q2_K.gguf",
    #     "download_url": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q2_K.gguf",
    #     "fallback": {
    #         "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    #         "local_path": os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0-gguf")
    #     },
    #     "config_template": "optimized",
    #     "config_override": {
    #         "n_gpu_layers": 1 if IS_CUDA else 0,
    #     },
    #     "description": "Large powerful MoE model (low quantization) (13GB)"
    # },
    
    # Llama models
    "tinyllama-1.1b": {
        "model_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "family": "llama",
        "local_path": os.path.join(MODELS_DIR, "tinyllama-1.1b-chat-v1.0-gguf"),
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "download_url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "config_template": "base",
        "description": "Very small model for limited hardware (600MB)"
    },
    
    # Qwen models
    # "qwen-2-1.5b": {
    #     "model_id": "QuantFactory/Qwen2-1.5B-Instruct-GGUF",
    #     "family": "qwen",
    #     "local_path": os.path.join(MODELS_DIR, "qwen2-1.5b-instruct-gguf"),
    #     "filename": "Qwen2-1.5B-Instruct.Q4_K_M.gguf",
    #     "download_url": "https://huggingface.co/QuantFactory/Qwen2-1.5B-Instruct-GGUF/resolve/main/Qwen2-1.5B-Instruct.Q4_K_M.gguf",
    #     "config_template": "base",
    #     "description": "Small but capable Chinese/English model (1.3GB)"
    # }
}

# Default model
DEFAULT_MODEL = "tinyllama-1.1b"
MODEL_CONFIG = AVAILABLE_MODELS[DEFAULT_MODEL]

# Get specific model settings from the active model config
MODEL_ID = MODEL_CONFIG["model_id"]
MODEL_FAMILY = MODEL_CONFIG["family"]
MODEL_TYPE = MODEL_FAMILIES[MODEL_FAMILY]["model_type"]
LOCAL_MODEL_PATH = MODEL_CONFIG["local_path"]
MODEL_FILENAME = MODEL_CONFIG["filename"]

# Build LLAMACPP_CONFIGS dynamically from templates and AVAILABLE_MODELS
LLAMACPP_CONFIGS = {
    "default": LLAMACPP_BASE_CONFIG.copy(),
    "fallback": {
        "temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
        "n_ctx": 2048,
        "n_batch": 256,
        "n_gpu_layers": 0,           # CPU only
        "verbose": True
    }
}

# Generate model-specific configurations
for model_name, model_info in AVAILABLE_MODELS.items():
    # Start with the appropriate template
    template_name = model_info.get("config_template", "base")
    if template_name == "optimized":
        config = LLAMACPP_OPTIMIZED_CONFIG.copy()
    else:
        config = LLAMACPP_BASE_CONFIG.copy()
    
    # Apply any model-specific overrides
    if "config_override" in model_info:
        config.update(model_info["config_override"])
    
    # Add to LLAMACPP_CONFIGS
    LLAMACPP_CONFIGS[model_name] = config

# CTransformers configurations
CTRANSFORMERS_CONFIGS = {
    "default": {
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': LLM_TEMPERATURE,
        'context_length': LLM_CONTEXT_LENGTH,
        'gpu_layers': LLM_GPU_LAYERS
    }
}

# Model-specific stop sequences for generation
STOP_SEQUENCES = {model_family: info["stop_words"] for model_family, info in MODEL_FAMILIES.items()}

# Prompt templates for different model types
PROMPT_TEMPLATES = {model_family: info["prompt_template"] for model_family, info in MODEL_FAMILIES.items()}

# Create directories if they don't exist
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VECTORDB_PATH, exist_ok=True)

# Create model directories
for model_name, config in AVAILABLE_MODELS.items():
    os.makedirs(config["local_path"], exist_ok=True) 