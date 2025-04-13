#!/usr/bin/env python3

import os
import torch
import platform
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

def main():
    # Path to the GGUF file
    model_path = "models/mistral-7b-instruct-v0.2-gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Check for GPU availability
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device count: {gpu_count}")
    if cuda_available:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check for Apple Silicon (M1/M2/M3)
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        print("Apple Silicon (M1/M2/M3) detected: Metal acceleration should be active")
        print("Note: torch.cuda.is_available() will show False, but Metal is being used")
    
    print(f"Loading model from {model_path}...")
    
    try:
        # Use either CUDA or Metal depending on platform
        use_gpu_acceleration = cuda_available or is_apple_silicon
        
        # Initialize the model with GPU acceleration if available
        llm = CTransformers(
            model=model_path,
            model_type="mistral",
            config={
                'max_new_tokens': 512,
                'temperature': 0.1,
                'context_length': 4096,
                'gpu_layers': 32 if use_gpu_acceleration else 0  # Use GPU if available
            }
        )
        
        # Context from our knowledge base
        context = """
        # Industrial Pump XYZ-123 Specifications

        ## Technical Specifications
        - Model: XYZ-123
        - Type: Centrifugal Pump
        - Flow Rate: 100-500 GPM
        - Head: 50-200 ft
        - Operating Temperature: 20°C to 80°C
        - NPSH Required: 10 ft
        - Maximum Working Pressure: 250 PSI
        - Motor: 50 HP, 3-phase, 460V
        - Weight: 750 lbs
        """
        
        # Create the prompt
        prompt = PromptTemplate.from_template(
            """<s>[INST] You are an industrial assistant that answers questions based on the provided context.
Answer the question based only on the context provided. If you don't know the answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question} [/INST]"""
        )
        
        # Generate the query 
        query = "What is the operating temperature of the XYZ-123 pump?"
        final_prompt = prompt.format(context=context, question=query)
        
        print(f"Query: {query}")
        print("Generating answer...")
        
        # Get the response
        response = llm.invoke(final_prompt)
        
        print(f"\nAnswer: {response}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 