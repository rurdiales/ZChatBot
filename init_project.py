#!/usr/bin/env python3
import os
import subprocess
import sys
import platform
import argparse
import shutil

def run_command(command, cwd=None):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, 
                              capture_output=True, cwd=cwd)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Initialize the Industrial RAG Chatbot project")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument("--force-venv", action="store_true", help="Force recreation of virtual environment")
    args = parser.parse_args()
    
    print("Starting Industrial RAG Chatbot initialization...")
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Create necessary directories
    for directory in ["models", "data", "knowledge"]:
        dir_path = os.path.join(current_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {directory}")
    
    if not args.skip_venv:
        # Create virtual environment
        print("\nSetting up virtual environment...")
        venv_path = os.path.join(current_dir, "venv")
        
        # Check if venv exists but is corrupted or if force recreation is enabled
        if args.force_venv and os.path.exists(venv_path):
            print("Removing existing virtual environment...")
            try:
                shutil.rmtree(venv_path)
                print("Existing virtual environment removed.")
            except Exception as e:
                print(f"Error removing virtual environment: {e}")
                return
        
        # Create new venv if it doesn't exist or was removed
        if not os.path.exists(venv_path):
            print("Creating new virtual environment...")
            if not run_command(f"{sys.executable} -m venv venv"):
                print("Failed to create virtual environment. Exiting.")
                return
            print("Virtual environment created successfully.")
        else:
            # Test if venv is working
            pip_cmd = os.path.join("venv", "bin", "pip") if platform.system() != "Windows" else os.path.join("venv", "Scripts", "pip")
            pip_path = os.path.join(current_dir, pip_cmd)
            
            if not os.path.exists(pip_path) or not run_command(f"{pip_path} --version"):
                print("Existing virtual environment appears to be corrupted. Use --force-venv to recreate it.")
                return
            print("Using existing virtual environment.")
        
        # Determine activation command based on platform
        if platform.system() == "Windows":
            activate_path = os.path.join(venv_path, "Scripts", "activate")
            pip_cmd = os.path.join(venv_path, "Scripts", "pip")
        else:
            activate_path = os.path.join(venv_path, "bin", "activate")
            pip_cmd = os.path.join(venv_path, "bin", "pip")
        
        # Verify pip exists
        if not os.path.exists(pip_cmd):
            print(f"Error: pip not found at {pip_cmd}")
            print("The virtual environment may be corrupted. Use --force-venv to recreate it.")
            return
        
        # Install dependencies
        print("\nInstalling dependencies...")
        if not run_command(f'"{pip_cmd}" install --upgrade pip'):
            print("Failed to upgrade pip. Continuing anyway...")
        
        requirements_path = os.path.join(current_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            if not run_command(f'"{pip_cmd}" install -r "{requirements_path}"'):
                print("Failed to install dependencies from requirements.txt.")
                
                # Try installing directly using setup.py as a fallback
                setup_path = os.path.join(current_dir, "setup.py")
                if os.path.exists(setup_path):
                    print("Trying to install via setup.py...")
                    if not run_command(f'"{pip_cmd}" install -e .'):
                        print("Failed to install dependencies. Please check your internet connection and try again.")
                        return
                else:
                    print("setup.py not found. Cannot install dependencies.")
                    return
        else:
            print("requirements.txt not found, trying to install via setup.py...")
            if not run_command(f'"{pip_cmd}" install -e .'):
                print("Failed to install dependencies. Please check your internet connection and try again.")
                return
        
        print("Dependencies installed successfully.")
    
    if not args.skip_model:
        # Download model
        model_dir = os.path.join(current_dir, "models/mistral-7b-instruct-v0.2-gguf")
        model_path = os.path.join(model_dir, "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        
        if os.path.exists(model_path):
            print(f"\nModel already exists at {model_path}")
        else:
            print(f"\nDownloading Mistral 7B model to {model_path}...")
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if curl is available
            if run_command("curl --version", cwd=current_dir):
                download_command = (
                    f"curl -L "
                    f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf "
                    f"-o \"{model_path}\""
                )
                
                if not run_command(download_command, cwd=current_dir):
                    print("Failed to download model. Please download it manually.")
                    print("See README.md for instructions.")
                else:
                    print("Model downloaded successfully!")
            else:
                # Fall back to Python for downloading
                print("curl not found, using Python to download...")
                try:
                    import requests
                    print("Downloading model... (this may take a while)")
                    url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
                    r = requests.get(url, stream=True)
                    total_size = int(r.headers.get('content-length', 0))
                    with open(model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print("Download complete.")
                except Exception as e:
                    print(f"Error downloading model: {e}")
                    print("Please download it manually. See README.md for instructions.")
    
    print("\n=== Setup Complete! ===")
    print(f"Project initialized in: {current_dir}")
    
    # Provide usage instructions
    if platform.system() == "Windows":
        print("\nTo activate the virtual environment:")
        print(f"{os.path.join('venv', 'Scripts', 'activate')}")
    else:
        print("\nTo activate the virtual environment:")
        print(f"source {os.path.join('venv', 'bin', 'activate')}")
    
    print("\nTo use the chatbot:")
    print("1. Process documents: python main.py --process")
    print("2. Start interactive mode: python main.py --interactive")
    print("\nEnjoy your Industrial RAG Chatbot!")

if __name__ == "__main__":
    main() 