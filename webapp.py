#!/usr/bin/env python3
import os
import sys
import platform
import torch
import gradio as gr
from src.chatbot import IndustrialChatbot
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL

# Detect hardware capabilities
IS_CUDA = torch.cuda.is_available()
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"
IS_WINDOWS = platform.system() == "Windows"

# Filter models based on hardware compatibility
def get_compatible_models():
    """Filter models based on hardware compatibility."""
    compatible_models = {}
    
    for model_name, model_info in AVAILABLE_MODELS.items():
        is_compatible = True
        compatibility_note = ""
        
        # Special handling for specific models
        if "mixtral" in model_name.lower() and not IS_CUDA and not (IS_APPLE_SILICON and "fallback" in model_info):
            is_compatible = False
            compatibility_note = "Requires CUDA GPU (too large for CPU-only)"
        
        # CPU-only models are compatible with everything
        if model_name.endswith("-cpu"):
            is_compatible = True
            compatibility_note = "CPU-only (will work everywhere but slower)"
        
        # Add model to compatible list with note
        if is_compatible:
            compatible_models[model_name] = model_info
            model_info["compatibility_note"] = compatibility_note if compatibility_note else "Compatible with current hardware"
        else:
            # Add to list but mark as incompatible
            compatible_models[model_name] = model_info
            model_info["compatibility_note"] = compatibility_note if compatibility_note else "May not be compatible"
            model_info["is_incompatible"] = True
    
    return compatible_models

# Get compatible models
COMPATIBLE_MODELS = get_compatible_models()

# Choose a good default model for this hardware
def get_best_default_model():
    """Choose the best default model for the current hardware."""
    if DEFAULT_MODEL in COMPATIBLE_MODELS and not COMPATIBLE_MODELS[DEFAULT_MODEL].get("is_incompatible", False):
        return DEFAULT_MODEL
    
    # If on Apple Silicon, prefer phi-3-mini or phi-3-mini-cpu
    if IS_APPLE_SILICON:
        if "phi-3-mini" in COMPATIBLE_MODELS and not COMPATIBLE_MODELS["phi-3-mini"].get("is_incompatible", False):
            return "phi-3-mini"
        elif "phi-3-mini-cpu" in COMPATIBLE_MODELS:
            return "phi-3-mini-cpu"
    
    # If CUDA available, prefer a larger model that's compatible
    if IS_CUDA:
        for model_name in ["phi-3-medium", "phi-3-mini", "mistral-7b", "zephyr-7b"]:
            if model_name in COMPATIBLE_MODELS and not COMPATIBLE_MODELS[model_name].get("is_incompatible", False):
                return model_name
    
    # Fallback to tinyllama which should work anywhere
    if "tinyllama-1.1b" in COMPATIBLE_MODELS:
        return "tinyllama-1.1b"
    
    # Last resort - just pick the first compatible model
    for model_name, model_info in COMPATIBLE_MODELS.items():
        if not model_info.get("is_incompatible", False):
            return model_name
    
    # If nothing is compatible, return the original default but with a warning
    print("WARNING: No compatible models found for this hardware!")
    return DEFAULT_MODEL

# Get the best default model for this hardware
BEST_DEFAULT_MODEL = get_best_default_model()

# Print hardware info
print(f"Hardware detection:")
print(f"- CUDA GPU: {'Available' if IS_CUDA else 'Not available'}")
if IS_CUDA:
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print(f"- Apple Silicon: {'Yes' if IS_APPLE_SILICON else 'No'}")
print(f"- Platform: {platform.system()} {platform.release()}")
print(f"- Python: {platform.python_version()}")
print(f"\nUsing default model: {BEST_DEFAULT_MODEL}")

# Initialize the chatbot with the best default model
chatbot = IndustrialChatbot(model_name=BEST_DEFAULT_MODEL)

def process_knowledge_base():
    """Process all documents in the knowledge base."""
    chatbot.process_knowledge_base()
    return "Knowledge base processed successfully!"

def clear_knowledge_base():
    """Clear the vector store."""
    chatbot.clear_knowledge_base()
    return "Knowledge base cleared successfully!"

def switch_model(model_name):
    """Switch to a different model."""
    if model_name not in AVAILABLE_MODELS:
        return f"Model {model_name} not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
    
    # Check if model is compatible
    if COMPATIBLE_MODELS.get(model_name, {}).get("is_incompatible", False):
        compatibility_note = COMPATIBLE_MODELS[model_name].get("compatibility_note", "Unknown compatibility issue")
        return f"⚠️ Warning: {model_name} may not be compatible with this hardware: {compatibility_note}. Attempting to load anyway..."
    
    global chatbot
    try:
        chatbot.switch_model(model_name)
        return f"Switched to model: {model_name}"
    except Exception as e:
        return f"Error switching to model {model_name}: {str(e)}"

def chat(message, history):
    """Generate response from the chatbot."""
    try:
        response = chatbot.ask(message)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}\n\nTry switching to a different model that's compatible with this hardware."

def get_model_info(model_name):
    """Get information about a specific model."""
    if model_name not in AVAILABLE_MODELS:
        return "Model not found"
    
    model_info = AVAILABLE_MODELS[model_name]
    info_text = f"Model: {model_name}\n"
    info_text += f"Description: {model_info.get('description', 'No description available')}\n"
    info_text += f"Family: {model_info.get('family', 'Unknown')}\n"
    
    # Add compatibility information
    if model_name in COMPATIBLE_MODELS:
        compatibility = COMPATIBLE_MODELS[model_name].get("compatibility_note", "Unknown")
        if COMPATIBLE_MODELS[model_name].get("is_incompatible", False):
            info_text += f"Compatibility: ⚠️ {compatibility}\n"
        else:
            info_text += f"Compatibility: ✅ {compatibility}\n"
    
    # Add file information if available
    if 'filename' in model_info:
        file_path = os.path.join(model_info['local_path'], model_info['filename'])
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            info_text += f"File: {model_info['filename']} ({size_mb:.2f} MB)\n"
            info_text += f"Status: Loaded"
        else:
            info_text += f"File: {model_info['filename']} (Not downloaded)\n"
            info_text += f"Status: Not available"
    
    return info_text

# Create the Gradio interface
with gr.Blocks(title="Industrial RAG Chatbot") as demo:
    gr.Markdown("# Industrial RAG Chatbot")
    gr.Markdown("Ask questions about your industrial documents using local LLMs.")
    
    # Show hardware info at the top
    hardware_info = f"Running on: {platform.system()} {'(Apple Silicon)' if IS_APPLE_SILICON else ''} {'with CUDA GPU' if IS_CUDA else 'without GPU'}"
    gr.Markdown(f"### {hardware_info}")
    
    with gr.Tab("Chat"):
        chatbot_interface = gr.ChatInterface(
            chat,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(placeholder="Ask a question about your documents...", container=False, scale=7),
            title="Industrial RAG Assistant",
        )
    
    with gr.Tab("Models"):
        with gr.Row():
            with gr.Column(scale=1):
                # Create dropdown with color-coding for compatibility
                model_choices = []
                for model_name in AVAILABLE_MODELS.keys():
                    if model_name in COMPATIBLE_MODELS:
                        if COMPATIBLE_MODELS[model_name].get("is_incompatible", False):
                            # Mark incompatible models
                            model_choices.append(f"⚠️ {model_name}")
                        else:
                            # Mark compatible models
                            model_choices.append(f"✅ {model_name}")
                    else:
                        model_choices.append(model_name)
                
                # Find the default model in the choices
                default_choice = None
                for choice in model_choices:
                    if choice.endswith(BEST_DEFAULT_MODEL) or choice.endswith(f" {BEST_DEFAULT_MODEL}"):
                        default_choice = choice
                        break
                
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_choice,
                    label="Select Model (✅ = Compatible with this hardware)"
                )
                switch_btn = gr.Button("Switch Model")
                model_info = gr.Textbox(label="Model Information", value=get_model_info(BEST_DEFAULT_MODEL), lines=10)
                
                # Function to clean model name (remove emoji prefix)
                def clean_model_name(display_name):
                    if display_name.startswith("✅ "):
                        return display_name[2:].strip()
                    if display_name.startswith("⚠️ "):
                        return display_name[2:].strip()
                    return display_name
                
                # Connect the button to switch model function with name cleaning
                switch_btn.click(
                    lambda name: switch_model(clean_model_name(name)), 
                    inputs=model_dropdown, 
                    outputs=model_info
                )
                
                # Update model info when dropdown changes with name cleaning
                model_dropdown.change(
                    lambda name: get_model_info(clean_model_name(name)), 
                    inputs=model_dropdown, 
                    outputs=model_info
                )
    
    with gr.Tab("Knowledge Base"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Process Documents")
                gr.Markdown("Process documents in the 'knowledge' directory to make them searchable.")
                process_btn = gr.Button("Process Knowledge Base")
                process_output = gr.Textbox(label="Process Output", lines=2)
                
                # Connect the process button
                process_btn.click(process_knowledge_base, inputs=None, outputs=process_output)
                
            with gr.Column(scale=1):
                gr.Markdown("### Clear Knowledge Base")
                gr.Markdown("⚠️ This will delete all document embeddings from the vector store.")
                clear_btn = gr.Button("Clear Knowledge Base")
                clear_output = gr.Textbox(label="Clear Output", lines=2)
                
                # Connect the clear button
                clear_btn.click(clear_knowledge_base, inputs=None, outputs=clear_output)
    
    with gr.Tab("System Info"):
        system_info = f"""
### Hardware Information
- **Platform**: {platform.system()} {platform.release()}
- **Python Version**: {platform.python_version()}
- **GPU Acceleration**: {'Available' if IS_CUDA else 'Not available'}
"""
        if IS_CUDA:
            system_info += f"- **GPU Model**: {torch.cuda.get_device_name(0)}\n"
            system_info += f"- **GPU Memory**: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n"
        
        if IS_APPLE_SILICON:
            system_info += "- **Apple Silicon**: Yes (Metal acceleration should be available)\n"
        
        system_info += "\n### Compatible Models\n"
        
        for model_name, model_info in COMPATIBLE_MODELS.items():
            if not model_info.get("is_incompatible", False):
                system_info += f"- ✅ **{model_name}**: {model_info.get('description', 'No description')}\n"
        
        system_info += "\n### Potentially Incompatible Models\n"
        
        for model_name, model_info in COMPATIBLE_MODELS.items():
            if model_info.get("is_incompatible", False):
                system_info += f"- ⚠️ **{model_name}**: {model_info.get('compatibility_note', 'Unknown issue')}\n"
        
        gr.Markdown(system_info)

# Launch the app
if __name__ == "__main__":
    # Add custom port and sharing options
    import argparse
    parser = argparse.ArgumentParser(description="Launch Industrial RAG Chatbot web interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name/IP to bind to")
    args = parser.parse_args()
    
    # Launch with the parsed arguments
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    ) 