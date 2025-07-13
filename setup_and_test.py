#!/usr/bin/env python3
"""
Setup and test script for Claude RAG system.

This script helps verify that all dependencies are installed correctly
and that API keys are properly configured.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'anthropic',
        'openai', 
        'numpy',
        'sklearn',
        'tiktoken',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required keys"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("Copy env_example.txt to .env and add your API keys")
        return False
    
    print("✅ .env file exists")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not anthropic_key or anthropic_key == 'your_anthropic_api_key_here':
        print("❌ ANTHROPIC_API_KEY not properly set")
        return False
    print("✅ ANTHROPIC_API_KEY is set")
    
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("❌ OPENAI_API_KEY not properly set")
        return False
    print("✅ OPENAI_API_KEY is set")
    
    return True

def test_api_connections():
    """Test basic API connectivity"""
    print("\nTesting API connections...")
    
    try:
        # Test OpenAI API
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple test embedding
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="test"
        )
        print("✅ OpenAI API connection successful")
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False
    
    try:
        # Test Anthropic API
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Simple test message
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("✅ Anthropic API connection successful")
        
    except Exception as e:
        print(f"❌ Anthropic API connection failed: {e}")
        return False
    
    return True

def test_rag_system():
    """Test basic RAG functionality"""
    print("\nTesting RAG system...")
    
    try:
        from claude_rag import ClaudeRAG
        
        # Initialize RAG
        rag = ClaudeRAG()
        print("✅ RAG system initialized")
        
        # Add a simple document
        test_doc = "The capital of France is Paris. It is known for the Eiffel Tower."
        rag.add_documents([test_doc], ["test_doc"])
        print("✅ Document added successfully")
        
        # Test query
        answer = rag.query("What is the capital of France?")
        if "Paris" in answer:
            print("✅ RAG query test successful")
            print(f"   Answer: {answer[:100]}...")
        else:
            print("❌ RAG query test failed - unexpected answer")
            return False
            
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False
    
    return True

def install_dependencies():
    """Install dependencies using pip"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main setup and test function"""
    print("🚀 Claude RAG System - Setup and Test")
    print("="*50)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        return
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n📦 Installing missing dependencies...")
        if install_dependencies():
            print("✅ All dependencies installed!")
        else:
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install -r requirements.txt")
            return
    
    # Check environment setup
    print("\n3. Checking environment setup...")
    if not check_env_file():
        print("\n📝 Setup instructions:")
        print("1. Copy env_example.txt to .env")
        print("2. Edit .env with your actual API keys")
        print("3. Run this script again")
        return
    
    # Test API connections
    print("\n4. Testing API connections...")
    if not test_api_connections():
        print("❌ API connection tests failed")
        print("Please check your API keys in .env file")
        return
    
    # Test RAG system
    print("\n5. Testing RAG system...")
    if not test_rag_system():
        return
    
    print("\n🎉 All tests passed! Your Claude RAG system is ready to use.")
    print("\nNext steps:")
    print("- Run: python example_usage.py (interactive example)")
    print("- Run: python demo_use_cases.py (demonstration of use cases)")
    print("- Import claude_rag in your own projects")

if __name__ == "__main__":
    main() 