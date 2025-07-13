#!/usr/bin/env python3
"""
Example usage of Claude RAG with local embeddings (no OpenAI API key needed).

This version uses sentence-transformers for local embeddings instead of OpenAI's API.
"""

import os
from claude_rag_no_openai import ClaudeRAGLocal

def main():
    # Check if Anthropic API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please create a .env file with your Anthropic API key:")
        print("ANTHROPIC_API_KEY=your_anthropic_api_key_here")
        return
    
    print("🚀 Initializing Claude RAG with Local Embeddings...")
    print("📦 This will download a small embedding model on first run (~90MB)")
    
    try:
        # Initialize RAG system (only needs Anthropic API key)
        rag = ClaudeRAGLocal()
        
        # Sample documents about AI/ML topics
        documents = [
            """
            Artificial Intelligence (AI) is a revolutionary field of computer science that focuses on 
            creating intelligent machines capable of performing tasks that typically require human 
            intelligence. These tasks include learning, reasoning, problem-solving, perception, and 
            language understanding. AI has applications across numerous industries including healthcare, 
            finance, transportation, and entertainment. Modern AI systems use machine learning 
            algorithms, neural networks, and deep learning to process vast amounts of data and make 
            intelligent decisions.
            """,
            """
            Machine Learning (ML) is a core subset of artificial intelligence that enables computers 
            to learn and improve from experience without being explicitly programmed for every task. 
            ML algorithms build mathematical models based on training data to make predictions or 
            decisions. There are three main types: supervised learning (learning from labeled examples), 
            unsupervised learning (finding patterns in unlabeled data), and reinforcement learning 
            (learning through trial and error with rewards). Popular ML techniques include decision 
            trees, support vector machines, neural networks, and ensemble methods.
            """,
            """
            Natural Language Processing (NLP) is an exciting interdisciplinary field that combines 
            computer science, artificial intelligence, and linguistics to help computers understand, 
            interpret, and generate human language. NLP enables machines to read text, hear speech, 
            and interpret meaning. Common NLP applications include language translation, sentiment 
            analysis, chatbots, voice assistants, and text summarization. Modern NLP relies heavily 
            on transformer architectures, attention mechanisms, and large language models trained 
            on massive text corpora.
            """,
            """
            Deep Learning is a specialized branch of machine learning inspired by the structure and 
            function of the human brain. It uses artificial neural networks with multiple layers 
            (hence "deep") to progressively extract higher-level features from raw input data. 
            Deep learning has achieved remarkable success in computer vision, speech recognition, 
            natural language processing, and game playing. Key architectures include convolutional 
            neural networks (CNNs) for image processing, recurrent neural networks (RNNs) for 
            sequential data, and transformers for language tasks.
            """
        ]
        
        sources = ["AI_Fundamentals", "ML_Guide", "NLP_Basics", "Deep_Learning_Overview"]
        
        # Add documents to RAG system
        print("📚 Adding documents to RAG system...")
        rag.add_documents(documents, sources)
        
        # Print system statistics
        stats = rag.get_stats()
        print(f"\n📊 RAG System Statistics:")
        print(f"   • Document chunks: {stats['total_documents']}")
        print(f"   • Embedding dimension: {stats['embedding_dimension']}")  
        print(f"   • Unique sources: {stats['unique_sources']}")
        print(f"   • Embedding model: {stats['embedding_model']}")
        
        # Interactive query session
        print("\n" + "="*60)
        print("🤖 CLAUDE RAG SYSTEM - Ask me about AI/ML!")
        print("="*60)
        print("💡 Try questions like:")
        print("   • What is artificial intelligence?")
        print("   • How does machine learning work?")
        print("   • What are the types of machine learning?")
        print("   • What is deep learning used for?")
        print("\nType 'quit' to exit")
        
        while True:
            try:
                query = input("\n🔍 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using Claude RAG! Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("🔎 Searching documents and generating response...")
                answer = rag.query(query)
                print(f"\n🤖 Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except ImportError as e:
        if "sentence-transformers" in str(e):
            print("❌ Missing dependency: sentence-transformers")
            print("📦 Install with: pip install sentence-transformers torch")
        else:
            print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("Make sure your Anthropic API key is set in the .env file")

if __name__ == "__main__":
    main() 