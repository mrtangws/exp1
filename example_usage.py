#!/usr/bin/env python3
"""
Example usage of the Claude RAG system.

Before running this script:
1. Install dependencies: pip install -r requirements.txt
2. Copy env_example.txt to .env and add your API keys
3. Run: python example_usage.py
"""

import os
from claude_rag import ClaudeRAG

def main():
    # Check if API keys are set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment")
        print("Please create a .env file with your API keys")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please create a .env file with your API keys")
        return
    
    print("Initializing Claude RAG system...")
    
    try:
        # Initialize RAG system
        rag = ClaudeRAG()
        
        # Sample documents about AI/ML topics
        documents = [
            """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that work and react like humans. Some of the activities 
            computers with artificial intelligence are designed for include speech recognition, 
            learning, planning, and problem solving. AI research has been highly successful 
            in developing effective techniques for solving a wide range of problems, from game 
            playing to medical diagnosis to logistics planning. Modern AI systems use machine 
            learning algorithms to improve their performance on specific tasks.
            """,
            """
            Machine Learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being 
            explicitly programmed. Machine learning focuses on the development of computer 
            programs that can access data and use it to learn for themselves. The process of 
            learning begins with observations or data, such as examples, direct experience, 
            or instruction, in order to look for patterns in data. There are three main types 
            of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
            """,
            """
            Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
            and artificial intelligence concerned with the interactions between computers and 
            human language, in particular how to program computers to process and analyze 
            large amounts of natural language data. NLP combines computational linguistics 
            with statistical, machine learning, and deep learning models. Common NLP tasks 
            include sentiment analysis, named entity recognition, machine translation, and 
            question answering systems.
            """,
            """
            Deep Learning is a subset of machine learning that uses neural networks with 
            multiple layers to model and understand complex patterns in data. Deep learning 
            algorithms attempt to mimic the way the human brain processes information through 
            artificial neural networks. These networks can learn to recognize patterns in 
            images, understand speech, translate languages, and even generate human-like text. 
            Popular deep learning frameworks include TensorFlow, PyTorch, and Keras.
            """
        ]
        
        sources = ["AI_Overview", "ML_Basics", "NLP_Introduction", "Deep_Learning"]
        
        # Add documents to RAG system
        print("Adding documents to RAG system...")
        rag.add_documents(documents, sources)
        
        # Print system statistics
        stats = rag.get_stats()
        print(f"\nRAG System Statistics:")
        print(f"- Total document chunks: {stats['total_documents']}")
        print(f"- Embedding dimension: {stats['embedding_dimension']}")
        print(f"- Unique sources: {stats['unique_sources']}")
        
        # Interactive query loop
        print("\n" + "="*60)
        print("CLAUDE RAG SYSTEM - Ready for queries!")
        print("="*60)
        print("Enter your questions (or 'quit' to exit)")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nSearching and generating response...")
            try:
                answer = rag.query(query)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("Make sure your API keys are correctly set in the .env file")

if __name__ == "__main__":
    main() 