#!/usr/bin/env python3
"""
Simple test script for the Prisma Scraper framework
"""
import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_environment():
    """Test the environment setup"""
    print("🧪 Testing Environment Setup")
    print("-" * 40)
    
    # Test Python version
    print(f"Python Version: {sys.version}")
    
    # Test environment variables
    print("\n📁 Environment Variables:")
    env_file = current_dir / ".env"
    if env_file.exists():
        print(f"✅ Found .env file at: {env_file}")
        with open(env_file, 'r') as f:
            content = f.read()
            if "MONGODB_URL" in content:
                print("✅ MongoDB configuration found")
            if "MODEL_NAME" in content:
                print("✅ Model configuration found")
    else:
        print("❌ .env file not found")
    
    # Test basic imports
    print("\n📦 Testing Basic Imports:")
    
    try:
        import os
        print("✅ os")
    except ImportError as e:
        print(f"❌ os: {e}")
    
    try:
        from datetime import datetime
        print("✅ datetime")
    except ImportError as e:
        print(f"❌ datetime: {e}")
    
    try:
        import json
        print("✅ json")
    except ImportError as e:
        print(f"❌ json: {e}")
    
    # Test optional imports
    print("\n🔧 Testing Optional Imports:")
    
    packages_to_test = [
        "pymongo",
        "torch", 
        "transformers",
        "langchain",
        "instructor",
        "openai",
        "pydantic",
        "loguru",
        "tenacity"
    ]
    
    for package in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
    
    print("\n" + "="*50)

def test_config():
    """Test configuration loading"""
    print("\n⚙️  Testing Configuration")
    print("-" * 40)
    
    try:
        # Try to load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
        
        # Check MongoDB configuration
        mongodb_url = os.getenv("MONGODB_URL")
        if mongodb_url:
            print("✅ MongoDB URL configured")
        else:
            print("❌ MongoDB URL not found")
        
        # Check model configuration
        model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small")
        print(f"✅ Model: {model_name}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("\n🔍 Testing Basic Functionality")
    print("-" * 40)
    
    # Test article input/output models without pydantic
    try:
        print("✅ Testing basic data structures...")
        
        # Simple article representation
        article_data = {
            "url": "https://example.com/test-article",
            "article": "This is a test article about political events.",
            "processed": False
        }
        print(f"✅ Article data structure: {len(article_data)} fields")
        
        # Simple bias classification
        bias_data = {
            "political": 1,
            "gender": 0,
            "cultural": 0,
            "ideology": 0
        }
        print(f"✅ Bias classification: {sum(bias_data.values())} biases detected")
        
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")

def main():
    """Run all tests"""
    print("🚀 Prisma Scraper Framework - Simple Test")
    print("=" * 50)
    
    test_environment()
    test_config()
    test_basic_functionality()
    
    print("\n🎯 Test Summary:")
    print("If you see mostly ✅ marks above, the basic setup is working!")
    print("❌ marks indicate missing dependencies or configuration issues.")
    print("\n📝 Next Steps:")
    print("1. Fix any ❌ issues shown above")
    print("2. Ensure MongoDB connection is available")
    print("3. Run the full framework once dependencies are resolved")

if __name__ == "__main__":
    main()
