#!/usr/bin/env python3
"""
Robust installation script for Claude RAG system.
Handles Python 3.13 compatibility issues and provides fallback options.
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def install_with_fallback():
    """Install packages with multiple fallback strategies"""
    
    print("🚀 Claude RAG Installation Script")
    print("="*50)
    
    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Strategy 1: Update build tools first
    print("\n📦 Strategy 1: Updating build tools...")
    if run_command("pip install --upgrade pip setuptools wheel", "Updating pip and setuptools"):
        if run_command("pip install -r requirements.txt", "Installing from requirements.txt"):
            print("✅ Installation successful with Strategy 1!")
            return True
    
    # Strategy 2: Individual package installation
    print("\n📦 Strategy 2: Installing packages individually...")
    packages = [
        ("anthropic", "Anthropic API client"),
        ("sentence-transformers", "Local embedding models"),
        ("torch", "PyTorch for embeddings"),
        ("numpy", "NumPy for numerical operations"),
        ("python-dotenv", "Environment variable management"),
        ("scikit-learn", "Machine learning utilities"),
        ("tiktoken", "Token counting"),
        ("pinecone-client", "Pinecone vector database integration")
    ]
    
    all_success = True
    for package, description in packages:
        if not run_command(f"pip install {package}", f"Installing {description}"):
            all_success = False
            # For non-critical packages, continue
            if package in ["scikit-learn", "tiktoken"]:
                print(f"⚠️  {package} failed but continuing (non-critical)")
            else:
                print(f"❌ {package} is critical - installation may be incomplete")
    
    if all_success:
        print("✅ Installation successful with Strategy 2!")
        return True
    
    # Strategy 3: Minimal installation
    print("\n📦 Strategy 3: Minimal installation (core packages only)...")
    critical_packages = [
        "anthropic",
        "sentence-transformers", 
        "numpy",
        "python-dotenv"
    ]
    
    minimal_success = True
    for package in critical_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            minimal_success = False
    
    if minimal_success:
        print("✅ Minimal installation successful!")
        print("⚠️  Note: Advanced features may be limited without scikit-learn and tiktoken")
        return True
    
    print("❌ All installation strategies failed")
    return False

def create_simple_rag():
    """Create a simplified version that works without optional dependencies"""
    simple_rag_code = '''"""
Simplified Claude RAG that works with minimal dependencies.
"""
import os
import re
import json
from typing import List, Dict, Any
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available, using basic math")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Error: Anthropic library required")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Error: OpenAI library required")

try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("Warning: python-dotenv not available")

class SimpleRAG:
    """Simplified RAG system with minimal dependencies"""
    
    def __init__(self):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic library is required")
        if not HAS_OPENAI:
            raise ImportError("openai library is required")
            
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str, source: str = ""):
        """Add a single document"""
        # Simple chunking by sentences
        sentences = re.split(r'(?<=[.!?])\\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > 1000:  # Simple length limit
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        for chunk in chunks:
            # Get embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embedding = response.data[0].embedding
            
            self.documents.append({
                'content': chunk,
                'source': source,
                'embedding': embedding
            })
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.documents:
            return "No documents available."
        
        # Get query embedding
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        query_embedding = response.data[0].embedding
        
        # Find most similar documents (simple dot product)
        similarities = []
        for doc in self.documents:
            if HAS_NUMPY:
                similarity = np.dot(query_embedding, doc['embedding'])
            else:
                # Simple dot product without numpy
                similarity = sum(a * b for a, b in zip(query_embedding, doc['embedding']))
            similarities.append(similarity)
        
        # Get top 3 most similar documents
        top_indices = sorted(range(len(similarities)), 
                           key=lambda i: similarities[i], reverse=True)[:3]
        
        context = "\\n\\n".join([self.documents[i]['content'] for i in top_indices])
        
        # Generate response with Claude
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
if __name__ == "__main__":
    rag = SimpleRAG()
    rag.add_document("The capital of France is Paris.", "geography")
    answer = rag.query("What is the capital of France?")
    print(answer)
'''
    
    with open("simple_claude_rag.py", "w") as f:
        f.write(simple_rag_code)
    print("✅ Created simple_claude_rag.py as fallback")

def main():
    """Main installation function"""
    
    success = install_with_fallback()
    
    if not success:
        print("\n🔧 Creating simplified version...")
        create_simple_rag()
        print("\n📝 Installation Summary:")
        print("❌ Full installation failed")
        print("✅ Created simple_claude_rag.py as alternative")
        print("\nTo use the simplified version:")
        print("1. Set up your API keys in environment variables")
        print("2. Run: python simple_claude_rag.py")
        return
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        if os.path.exists('env_example.txt'):
            run_command("cp env_example.txt .env", "Creating .env file")
            print("📝 Please edit .env with your actual API keys")
        else:
            print("⚠️  Please create .env file with your API keys")
    
    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Edit .env with your Anthropic API key")
    print("2. Try: python3 example_local_rag.py (local embeddings, no OpenAI needed)")
    print("3. Or: python3 example_usage.py (if you have OpenAI API key)")
    print("4. Run: python3 setup_and_test.py (to test full system)")

if __name__ == "__main__":
    main() 