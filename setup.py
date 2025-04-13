from setuptools import setup, find_packages
import os
import subprocess
import sys
from setuptools.command.develop import develop
from setuptools.command.install import install

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        self.setup_project()

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        self.setup_project()

    def setup_project(self):
        print("Setting up the Industrial RAG Chatbot project...")
        
        # Create necessary directories
        for directory in ["models", "data", "knowledge"]:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Download GGUF model
        model_dir = "models/mistral-7b-instruct-v0.2-gguf"
        model_path = f"{model_dir}/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        
        if not os.path.exists(model_path):
            print(f"Downloading Mistral 7B model to {model_path}...")
            os.makedirs(model_dir, exist_ok=True)
            
            try:
                subprocess.check_call([
                    "curl", "-L", 
                    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    "-o", model_path
                ])
                print("Model downloaded successfully!")
            except subprocess.CalledProcessError:
                print("Failed to download model. Please download it manually.")
                print("See README.md for instructions.")
        else:
            print(f"Model already exists at {model_path}")
        
        print("\nSetup complete! You can now use the chatbot.")
        print("To process documents: python main.py --process")
        print("To start interactive mode: python main.py --interactive")

setup(
    name="industrial_chatbot",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "langchain",
        "langchain_community",
        "ctransformers",
        "llama-cpp-python",
        "transformers",
        "accelerate",
        "bitsandbytes",
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
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
) 