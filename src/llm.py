import os
import platform
import time
import subprocess
from typing import List, Dict, Any, Optional
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.config import (
    AVAILABLE_MODELS, 
    DEFAULT_MODEL, 
    MAX_NEW_TOKENS, 
    LLM_TEMPERATURE, 
    LLM_CONTEXT_LENGTH, 
    LLM_GPU_LAYERS, 
    LLM_VERBOSE, 
    STOP_SEQUENCES,
    PROMPT_TEMPLATES,
    LLAMACPP_CONFIGS,
    CTRANSFORMERS_CONFIGS,
    MODEL_FAMILIES
)
import torch

# Import llama-cpp-python for Phi-3 support
try:
    from langchain_community.llms import LlamaCpp
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

class LocalLLM:
    """Interface to the local LLM."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        if model_name not in AVAILABLE_MODELS:
            print(f"Model {model_name} not found. Using default model {DEFAULT_MODEL}.")
            self.model_name = DEFAULT_MODEL
            
        self.model_config = AVAILABLE_MODELS[self.model_name]
        self.model_id = self.model_config["model_id"]
        self.model_path = self.model_config["local_path"]
        self.family = self.model_config["family"]
        self.model_type = MODEL_FAMILIES[self.family]["model_type"]
        self.model_filename = self.model_config["filename"]
        self.model = self._load_model()
    
    def _download_model(self, gguf_file: str) -> bool:
        """Download the model if not available."""
        try:
            print(f"Model file not found. Downloading {self.model_name} model...")
            download_url = self.model_config["download_url"]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(gguf_file), exist_ok=True)
            
            print("=" * 80)
            print(f"IMPORTANT: Automatic download might not work due to Hugging Face limitations.")
            print(f"Please manually download the model using this command in your terminal:")
            print()
            print(f"curl -L {download_url} -o {gguf_file}")
            print()
            print("Or download directly from your browser:")
            print(f"{download_url}")
            print(f"And save it to: {gguf_file}")
            print("=" * 80)
            
            # Try to download anyway
            try:
                subprocess.run(["curl", "-L", download_url, "-o", gguf_file], check=True, timeout=30)
                
                # Verify file size to confirm it's a real model (models are several GB, not a few bytes)
                if os.path.exists(gguf_file) and os.path.getsize(gguf_file) > 1_000_000:  # > 1MB
                    print(f"Model downloaded successfully to {gguf_file}")
                    return True
                else:
                    print("Download failed or incomplete. Please download manually using the instructions above.")
                    return False
            except Exception as e:
                print(f"Download attempt failed: {e}")
                return False
        except Exception as e:
            print(f"Error setting up download: {e}")
            print(f"Please download manually from {download_url} to {gguf_file}")
            return False
    
    def _load_model(self):
        """Load the model from local path."""
        # Better GPU detection including Apple Silicon
        is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            print(f"NVIDIA GPU acceleration: Enabled")
            print(f"Using: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        elif is_apple_silicon:
            print(f"Apple Silicon (M1/M2/M3) detected: Metal acceleration should be active")
            print(f"Note: torch.cuda.is_available() will show False, but Metal is being used")
        else:
            print(f"GPU acceleration: Disabled (CPU-only mode)")
        
        # If this is Mixtral on Apple Silicon, check if we should use the fallback model
        if "mixtral" in self.model_name.lower() and is_apple_silicon:
            print("Mixtral detected on Apple Silicon - this model may require more RAM than available")
            print("Will attempt to load but may fall back to TinyLlama if needed")
            
            # Check for fallback model
            if "fallback" in self.model_config:
                fallback_info = self.model_config["fallback"]
                fallback_path = fallback_info["local_path"]
                fallback_file = os.path.join(fallback_path, fallback_info["filename"])
                if os.path.exists(fallback_file):
                    print(f"Fallback model found at {fallback_file}")
        
        # Configure based on hardware
        use_gpu_acceleration = has_cuda or is_apple_silicon

        # The path for the GGUF file
        gguf_file = os.path.join(self.model_path, self.model_filename)
        
        if not os.path.exists(gguf_file):
            if not self._download_model(gguf_file):
                print(f"Please download the {self.model_name} model with the following command:")
                print(f"mkdir -p {self.model_path}")
                print(f"curl -L {self.model_config['download_url']} -o {gguf_file}")
                return None
        
        print(f"Loading model: {self.model_name} from {gguf_file}")
        load_start_time = time.time()
        
        # Use LlamaCpp for Phi-3 models
        if self.model_type == "phi" and LLAMACPP_AVAILABLE:
            try:
                # Use LlamaCpp instead of CTransformers for Phi-3
                print(f"Using LlamaCpp for {self.model_name} model...")
                
                # Get stop sequences for the model type
                stop_seqs = STOP_SEQUENCES.get(self.family, [])
                
                # Get model-specific configuration or use default
                model_config = LLAMACPP_CONFIGS.get(self.model_name, LLAMACPP_CONFIGS["default"]).copy()
                model_config["model_path"] = gguf_file
                
                # Adjust GPU layers based on hardware
                if not use_gpu_acceleration:
                    model_config["n_gpu_layers"] = 0
                
                # Add stop sequences if defined
                if stop_seqs:
                    model_config["stop"] = stop_seqs
                
                model = LlamaCpp(**model_config)
                load_time = time.time() - load_start_time
                print(f"Model loaded with LlamaCpp in {load_time:.2f} seconds")
                
                return model
            except Exception as e:
                print(f"Error loading with LlamaCpp: {e}")
                print("Falling back to CTransformers...")
        
        # Also use LlamaCpp for Mixtral models
        if "mixtral" in self.model_name.lower() and LLAMACPP_AVAILABLE:
            try:
                print(f"Using LlamaCpp for {self.model_name} model...")
                print(f"File exists check: {os.path.exists(gguf_file)}")
                if os.path.exists(gguf_file):
                    print(f"File size: {os.path.getsize(gguf_file) / (1024*1024*1024):.2f} GB")
                
                # Get stop sequences for the model type
                stop_seqs = STOP_SEQUENCES.get(self.family, [])
                
                # Get Mixtral-specific configuration from config.py
                model_config = LLAMACPP_CONFIGS.get(self.model_name, LLAMACPP_CONFIGS["default"]).copy()
                model_config["model_path"] = gguf_file
                
                # Add stop sequences if defined
                if stop_seqs:
                    model_config["stop"] = stop_seqs
                
                print(f"Loading Mixtral with parameters: {model_config}")
                model = LlamaCpp(**model_config)
                load_time = time.time() - load_start_time
                print(f"Mixtral model loaded with LlamaCpp in {load_time:.2f} seconds")
                
                return model
            except Exception as e:
                print(f"Error loading Mixtral with LlamaCpp: {str(e)}")
                print("Attempting to load fallback model for Mixtral...")
                
                # Try to use the fallback model for Mixtral
                if "fallback" in self.model_config:
                    fallback_info = self.model_config["fallback"]
                    fallback_path = fallback_info["local_path"]
                    fallback_file = os.path.join(fallback_path, fallback_info["filename"])
                    
                    if os.path.exists(fallback_file):
                        print(f"Loading fallback model: {fallback_file}")
                        try:
                            # Use fallback configuration from config.py
                            fallback_config = LLAMACPP_CONFIGS["fallback"].copy()
                            fallback_config["model_path"] = fallback_file
                            
                            model = LlamaCpp(**fallback_config)
                            load_time = time.time() - load_start_time
                            print(f"Fallback model loaded in {load_time:.2f} seconds")
                            return model
                        except Exception as e2:
                            print(f"Error loading fallback model: {str(e2)}")
                    else:
                        print(f"Fallback model not found at {fallback_file}")
                
                print("Trying CTransformers as last resort...")
        
        # List of model types to try (original first, then fallbacks)
        model_types_to_try = [self.model_type]
        
        # For phi model, try several different model types as fallbacks
        if self.model_type == "phi":
            model_types_to_try.append("gpt2")  # Some Phi-3 models work with gpt2 architecture
            model_types_to_try.append("gptj")  # Try gptj architecture
            model_types_to_try.append("starcoder")  # Try starcoder architecture
            model_types_to_try.append("llama")  # Then try llama
            model_types_to_try.append("mistral")  # Finally try mistral
        
        # Get CTransformers config from config.py
        ct_config = CTRANSFORMERS_CONFIGS.get("default", {}).copy()
        
        # Adjust GPU layers based on hardware
        if not use_gpu_acceleration:
            ct_config["gpu_layers"] = 0
        
        # Try each model type
        last_error = None
        for model_type in model_types_to_try:
            try:
                print(f"Trying to load with model_type={model_type}...")
                
                # Directly use CTransformers with the GGUF file
                model = CTransformers(
                    model=gguf_file,
                    model_type=model_type,
                    config=ct_config
                )
                
                load_time = time.time() - load_start_time
                print(f"Model loaded in {load_time:.2f} seconds with model_type={model_type}")
                return model
                
            except Exception as e:
                print(f"Error loading with model_type={model_type}: {e}")
                last_error = e
        
        # If we got here, all model types failed
        print(f"Failed to load model with all attempted model types")
        if is_apple_silicon:
            print("For Apple Silicon Macs, make sure ctransformers is compiled with Metal support:")
            print("pip uninstall ctransformers --yes")
            print("CT_METAL=1 pip install ctransformers --no-binary ctransformers")
        return None
    
    def generate_answer(self, query: str, context_docs: Optional[List[Document]] = None) -> str:
        """Generate an answer to the query using the LLM and optional context."""
        if self.model is None:
            return f"Model {self.model_name} not loaded. Please download the model first using the instructions provided above."
        
        # Create context from documents
        context = ""
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Get the prompt template for the model family
        prompt_template = PROMPT_TEMPLATES.get(self.family, PROMPT_TEMPLATES["mistral"])
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Generate the response using the modern pattern (prompt | llm)
        try:
            # Time the response generation
            gen_start_time = time.time()
            
            # Create a runnable sequence (prompt | llm)
            runnable = prompt | self.model
            
            # Invoke the runnable with the input values
            response = runnable.invoke({"context": context, "query": query})
            
            gen_time = time.time() - gen_start_time
            print(f"Response generated in {gen_time:.2f} seconds with model {self.model_name}")
            
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating a response." 