#!/usr/bin/env python3
import os
import sys
import platform
import torch
import gradio as gr
from dotenv import load_dotenv
from src.chatbot import IndustrialChatbot
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL

# Load environment variables from .env file
load_dotenv()

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
        
        # Special handling for specific model families and sizes
        model_family = model_info.get("family", "")
        
        # Check for specific incompatible conditions
        if model_family == "mistral" and "16k" in model_name and not IS_CUDA and not IS_APPLE_SILICON:
            is_compatible = False
            compatibility_note = "Large context models require GPU or Apple Silicon"
            
        # OpenAI API models are always compatible
        elif model_family == "openai":
            is_compatible = True
            compatibility_note = "API model (requires API key)"
        
        # TinyLlama is compatible with everything
        elif "tinyllama" in model_name.lower():
            is_compatible = True
            compatibility_note = "Compatible with all hardware"
            
        # Phi-3 models are compatible with most hardware
        elif model_family == "phi" and IS_APPLE_SILICON:
            compatibility_note = "Optimized for Apple Silicon"
        elif model_family == "phi" and IS_CUDA:
            compatibility_note = "Optimized for CUDA GPU"
        elif model_family == "phi":
            compatibility_note = "Compatible with CPU but slower"
            
        # Add model to compatible list with note
        if is_compatible:
            compatible_models[model_name] = model_info.copy()
            compatible_models[model_name]["compatibility_note"] = compatibility_note if compatibility_note else "Compatible with current hardware"
        else:
            # Add to list but mark as incompatible
            compatible_models[model_name] = model_info.copy()
            compatible_models[model_name]["compatibility_note"] = compatibility_note if compatibility_note else "May not be compatible"
            compatible_models[model_name]["is_incompatible"] = True
    
    return compatible_models

# Get compatible models
COMPATIBLE_MODELS = get_compatible_models()

# Choose a good default model for this hardware
def get_best_default_model():
    """Choose the best default model for the current hardware."""
    if DEFAULT_MODEL in COMPATIBLE_MODELS and not COMPATIBLE_MODELS[DEFAULT_MODEL].get("is_incompatible", False):
        return DEFAULT_MODEL
    
    # If on Apple Silicon, prefer phi-3 models
    if IS_APPLE_SILICON:
        if "phi-3-mini-4k-instruct" in COMPATIBLE_MODELS and not COMPATIBLE_MODELS["phi-3-mini-4k-instruct"].get("is_incompatible", False):
            return "phi-3-mini-4k-instruct"
    
    # If CUDA available, prefer a larger model that's compatible
    if IS_CUDA:
        for model_name in ["mistral-7b-instruct-16k", "phi-3-mini-4k-instruct"]:
            if model_name in COMPATIBLE_MODELS and not COMPATIBLE_MODELS[model_name].get("is_incompatible", False):
                return model_name
    
    # If OpenAI API key is set, prefer gpt-3.5-turbo
    if os.environ.get("OPENAI_API_KEY") and "gpt-3.5-turbo" in COMPATIBLE_MODELS:
        return "gpt-3.5-turbo"
    
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
    
    # Start building HTML content
    html = "<div style='max-width: 800px;'>"
    html += f"<h3>{model_name}</h3>"
    html += f"<ul><li><strong>Description:</strong> {model_info.get('description', 'No description available')}</li>"
    html += f"<li><strong>Family:</strong> {model_info.get('family', 'Unknown')}</li>"
    
    # Add HuggingFace link for local models
    if "model_id" in model_info and model_info.get("family") != "openai":
        model_id = model_info["model_id"]
        if "/" in model_id:  # It's a HuggingFace model
            html += f"<li><strong>HuggingFace:</strong> <a href='https://huggingface.co/{model_id}' target='_blank'>https://huggingface.co/{model_id}</a></li>"
    
    # Add compatibility information
    if model_name in COMPATIBLE_MODELS:
        compatibility = COMPATIBLE_MODELS[model_name].get("compatibility_note", "Unknown")
        if COMPATIBLE_MODELS[model_name].get("is_incompatible", False):
            html += f"<li><strong>Compatibility:</strong> ⚠️ {compatibility}</li>"
        else:
            html += f"<li><strong>Compatibility:</strong> ✅ {compatibility}</li>"
    
    # Add file information if available
    if 'filename' in model_info:
        file_path = os.path.join(model_info['local_path'], model_info['filename'])
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            html += f"<li><strong>File:</strong> {model_info['filename']} ({size_mb:.2f} MB)</li>"
            html += f"<li><strong>Status:</strong> <span style='color:limegreen'>Loaded</span></li>"
        else:
            html += f"<li><strong>File:</strong> {model_info['filename']} (Not downloaded)</li>"
            html += f"<li><strong>Status:</strong> <span style='color:red'>Not available</span></li>"
            # Add download instructions
            html += f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
            html += f"<li><strong>Download Command:</strong></li>"
            html += f"<code>curl -L {model_info.get('download_url', '')} -o {file_path}</code>"
            html += f"</div>"
    
    html += "</div>"
    return html

# Add a function to handle file uploads
def upload_and_process_file(files):
    """Upload files to the knowledge directory and process them."""
    if not files:
        return "No files uploaded."
    
    try:
        results = []
        for file in files:
            # Get the filename from the path (handle both string paths and file objects)
            if isinstance(file, str):
                source_path = file
                filename = os.path.basename(file)
            else:
                source_path = file.name
                filename = os.path.basename(source_path)
            
            # Create the target path in the knowledge directory
            target_path = os.path.join("knowledge", filename)
            
            # Copy the file to the knowledge directory
            import shutil
            shutil.copy2(source_path, target_path)
            
            results.append(f"Uploaded: {filename}")
        
        # Process the knowledge base to include the new files
        chatbot.process_knowledge_base()
        results.append("\nKnowledge base processed with new files!")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error uploading files: {str(e)}"

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
                # Display the active model
                active_model_info = gr.HTML(
                    f"""
                    <div style='padding: 15px; 
                                background-color: rgba(0, 102, 204, 0.2); 
                                border: 2px solid rgba(0, 102, 204, 0.7);
                                border-radius: 8px; 
                                margin-bottom: 15px;
                                color: var(--body-text-color, #0066cc)'>
                    
                        <h3 style='margin-top: 0; color: var(--body-text-color, #0066cc); font-weight: bold'>
                            ⚡ Currently Active Model: {BEST_DEFAULT_MODEL}
                        </h3>
                        <p style='margin-bottom: 0; color: var(--body-text-color, #333333)'>
                            This is the model currently being used to answer questions.
                        </p>
                    </div>
                    """,
                    label="Active Model"
                )
                
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
                model_info = gr.HTML(get_model_info(BEST_DEFAULT_MODEL))
                
                # Function to clean model name (remove emoji prefix)
                def clean_model_name(display_name):
                    if display_name.startswith("✅ "):
                        return display_name[2:].strip()
                    if display_name.startswith("⚠️ "):
                        return display_name[2:].strip()
                    return display_name
                
                # Function to switch model and update both info panels
                def switch_and_update_info(model_name):
                    clean_name = clean_model_name(model_name)
                    
                    # Switch the model first
                    switch_result = switch_model(clean_name)
                    
                    # Get model information regardless of whether switch succeeded
                    model_information = get_model_info(clean_name)
                    
                    # Update active model banner - make it dark mode compatible
                    active_banner = f"""
                    <div style='padding: 15px; 
                                background-color: rgba(0, 102, 204, 0.2); 
                                border: 2px solid rgba(0, 102, 204, 0.7);
                                border-radius: 8px; 
                                margin-bottom: 15px;
                                color: var(--body-text-color, #0066cc)'>
                    
                       <h3 style='margin-top: 0; color: var(--body-text-color, #0066cc); font-weight: bold'>
                            ⚡ Currently Active Model: {clean_name if switch_result.startswith("Switched to model:") else chatbot.model_name}
                       </h3>
                       <p style='margin-bottom: 0; color: var(--body-text-color, #333333)'>
                           This is the model currently being used to answer questions.
                       </p>
                       
                       {f"<p style='color: #ff4444; margin-top: 10px; font-weight: bold'>Warning: {switch_result}</p>" 
                         if "Error" in switch_result or "not compatible" in switch_result else ""}
                    </div>
                    """
                    
                    return [model_information, active_banner]
                
                # Connect the button to the combined function
                switch_btn.click(
                    switch_and_update_info, 
                    inputs=model_dropdown, 
                    outputs=[model_info, active_model_info]
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
                gr.Markdown("### Upload Documents")
                gr.Markdown("Upload documents to add them to the knowledge base.")
                file_upload = gr.File(
                    file_count="multiple",
                    label="Upload Files",
                    file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
                )
                upload_btn = gr.Button("Upload and Process Files")
                upload_output = gr.Textbox(label="Upload Results", lines=5)
                
                # Connect the upload button
                upload_btn.click(upload_and_process_file, inputs=file_upload, outputs=upload_output)
                
            with gr.Column(scale=1):
                gr.Markdown("### Process Documents")
                gr.Markdown("Process existing documents in the 'knowledge' directory to make them searchable.")
                process_btn = gr.Button("Process Knowledge Base")
                process_output = gr.Textbox(label="Process Output", lines=2)
                
                # Connect the process button
                process_btn.click(process_knowledge_base, inputs=None, outputs=process_output)
                
        with gr.Row():
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
                # Add HuggingFace link for local models
                hf_link = ""
                if "model_id" in model_info and model_info.get("family") != "openai":
                    model_id = model_info["model_id"]
                    if "/" in model_id:  # It's a HuggingFace model
                        hf_link = f" - [HuggingFace](https://huggingface.co/{model_id})"
                
                system_info += f"- ✅ **{model_name}**: {model_info.get('description', 'No description')} ({model_info.get('compatibility_note', '')}){hf_link}\n"
        
        system_info += "\n### Potentially Incompatible Models\n"
        
        for model_name, model_info in COMPATIBLE_MODELS.items():
            if model_info.get("is_incompatible", False):
                # Add HuggingFace link for local models
                hf_link = ""
                if "model_id" in model_info and model_info.get("family") != "openai":
                    model_id = model_info["model_id"]
                    if "/" in model_id:  # It's a HuggingFace model
                        hf_link = f" - [HuggingFace](https://huggingface.co/{model_id})"
                
                system_info += f"- ⚠️ **{model_name}**: {model_info.get('compatibility_note', 'Unknown issue')}{hf_link}\n"
        
        system_info += """
### Multilingual Support
This chatbot has the following language capabilities:

- **Document Processing**: Fully multilingual support for documents in various languages
- **OCR Language Detection**: Automatic language detection for scanned documents
- **Query Processing**: Currently optimized for English queries
- **Bidirectional Search**: Limited support for non-English queries (experimental)

For best results:
- Upload documents in any language (Spanish, English, etc.)
- Ask questions in English for most reliable answers
- Non-English queries may have reduced retrieval accuracy
"""
        
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