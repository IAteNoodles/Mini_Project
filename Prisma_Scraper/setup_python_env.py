#!/usr/bin/env python3
"""
Python Environment Setup for Local Model Support
"""
import subprocess
import sys
import platform
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

def check_python_version():
    """Check current Python version"""
    version = sys.version_info
    print(f"🐍 Current Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 11 and version.minor <= 12:
        print("✅ Python version is compatible with PyTorch")
        return True
    elif version.major == 3 and version.minor == 13:
        print("⚠️  Python 3.13 (alpha) may have compatibility issues")
        print("💡 Recommended: Use Python 3.11 or 3.12 for best compatibility")
        return False
    else:
        print("❌ Python version not optimal for PyTorch")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"🔧 CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_python_311():
    """Instructions for setting up Python 3.11"""
    print("\n🔧 SETTING UP PYTHON 3.11 FOR OPTIMAL COMPATIBILITY")
    print("=" * 60)
    
    print("\n📝 OPTION 1: Using Anaconda/Miniconda (Recommended)")
    print("1. Download Miniconda: https://docs.conda.io/en/latest/miniconda.html")
    print("2. Install Miniconda")
    print("3. Create new environment:")
    print("   conda create -n prisma_llm python=3.11")
    print("4. Activate environment:")
    print("   conda activate prisma_llm")
    print("5. Install PyTorch with CUDA:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("6. Install other requirements:")
    print("   pip install transformers accelerate bitsandbytes pymongo python-dotenv")
    
    print("\n📝 OPTION 2: Using pyenv (Windows)")
    print("1. Install pyenv-win: https://github.com/pyenv-win/pyenv-win")
    print("2. Install Python 3.11:")
    print("   pyenv install 3.11.9")
    print("3. Set local version:")
    print("   pyenv local 3.11.9")
    print("4. Create virtual environment:")
    print("   python -m venv .venv_311")
    print("5. Activate and install packages:")
    print("   .venv_311\\Scripts\\activate")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n📝 OPTION 3: Docker (Most Reliable)")
    print("1. Create Dockerfile with Python 3.11 and CUDA support")
    print("2. Use pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel base image")

def install_dependencies_current_env():
    """Try to install dependencies in current environment"""
    print("\n🔧 ATTEMPTING TO FIX CURRENT ENVIRONMENT")
    print("=" * 50)
    
    commands = [
        ("pip uninstall torch torchvision torchaudio -y", "Uninstalling existing PyTorch"),
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch with CUDA 11.8"),
        ("pip install transformers==4.35.0", "Installing Transformers"),
        ("pip install accelerate==0.24.0", "Installing Accelerate"),
        ("pip install bitsandbytes==0.41.1", "Installing BitsAndBytes"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n📊 Installation Results: {success_count}/{len(commands)} successful")
    return success_count == len(commands)

def create_conda_environment():
    """Create a new conda environment with Python 3.11"""
    print("\n🔧 CREATING NEW CONDA ENVIRONMENT")
    print("=" * 40)
    
    commands = [
        ("conda create -n prisma_llm python=3.11 -y", "Creating conda environment"),
        ("conda activate prisma_llm && conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y", "Installing PyTorch"),
        ("conda activate prisma_llm && pip install transformers accelerate bitsandbytes pymongo python-dotenv loguru tenacity", "Installing Python packages"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
    
    if success_count == len(commands):
        print("\n✅ Conda environment created successfully!")
        print("🚀 To use the new environment:")
        print("   conda activate prisma_llm")
        print("   python classifier/local_model_summarizer.py")
        return True
    else:
        print("\n❌ Failed to create conda environment")
        return False

def test_environment():
    """Test if the environment can load PyTorch"""
    print("\n🧪 TESTING CURRENT ENVIRONMENT")
    print("=" * 35)
    
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"🔧 CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available")
        
        # Test model loading
        from transformers import AutoTokenizer
        print("✅ Transformers available")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Python Environment Setup for Local LLM")
    print("🤖 Mistral 7B Model Support")
    print("=" * 50)
    
    # Check current environment
    python_ok = check_python_version()
    
    print("\n🧪 Testing current environment...")
    if test_environment():
        print("\n🎉 CURRENT ENVIRONMENT IS WORKING!")
        print("✅ You can use the local model now")
        return
    
    print("\n❌ Current environment has issues")
    
    # Provide options
    print("\n🔧 SETUP OPTIONS:")
    print("1. Fix current environment (may not work with Python 3.13)")
    print("2. Create new Conda environment with Python 3.11 (recommended)")
    print("3. Manual setup instructions")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if install_dependencies_current_env():
            test_environment()
        else:
            print("\n❌ Failed to fix current environment")
            print("💡 Try option 2 (Conda) or 3 (Manual setup)")
    
    elif choice == "2":
        # Check if conda is available
        if run_command("conda --version", "Checking Conda"):
            create_conda_environment()
        else:
            print("❌ Conda not found. Please install Anaconda/Miniconda first")
            print("📥 Download: https://docs.conda.io/en/latest/miniconda.html")
    
    elif choice == "3":
        setup_python_311()
    
    elif choice == "4":
        print("👋 Exiting setup")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
