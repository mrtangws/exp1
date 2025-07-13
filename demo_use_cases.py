#!/usr/bin/env python3
"""
Demonstration of specific RAG use cases with Claude API.

This script shows practical applications of the Claude RAG system
for different types of document collections and query patterns.
"""

import os
from claude_rag import ClaudeRAG

def demo_tech_documentation():
    """Demo: Technical documentation Q&A"""
    print("\n" + "="*60)
    print("DEMO 1: Technical Documentation Q&A")
    print("="*60)
    
    rag = ClaudeRAG()
    
    # Sample technical documentation
    tech_docs = [
        """
        API Authentication
        
        Our REST API uses OAuth 2.0 for authentication. To access protected endpoints,
        you need to include a Bearer token in the Authorization header. 
        
        Steps to authenticate:
        1. Register your application to get client_id and client_secret
        2. Make a POST request to /oauth/token with your credentials
        3. Include the access_token in subsequent requests as: 
           Authorization: Bearer YOUR_ACCESS_TOKEN
        
        Token lifetime is 3600 seconds (1 hour). Refresh tokens are valid for 30 days.
        """,
        """
        Rate Limiting
        
        API requests are rate limited to prevent abuse:
        - Free tier: 100 requests per hour
        - Pro tier: 1000 requests per hour  
        - Enterprise: 10000 requests per hour
        
        Rate limit headers are included in responses:
        - X-RateLimit-Limit: Maximum requests allowed
        - X-RateLimit-Remaining: Requests remaining in current window
        - X-RateLimit-Reset: Unix timestamp when limit resets
        
        When rate limit is exceeded, API returns HTTP 429 status code.
        """,
        """
        Error Handling
        
        API errors follow HTTP status codes and include detailed error messages:
        
        Common error codes:
        - 400 Bad Request: Invalid request parameters
        - 401 Unauthorized: Missing or invalid authentication  
        - 403 Forbidden: Insufficient permissions
        - 404 Not Found: Resource does not exist
        - 429 Too Many Requests: Rate limit exceeded
        - 500 Internal Server Error: Server-side error
        
        Error response format:
        {
          "error": "error_code",
          "message": "Human readable error description",
          "details": {...}
        }
        """
    ]
    
    sources = ["auth_docs", "rate_limit_docs", "error_docs"]
    rag.add_documents(tech_docs, sources)
    
    queries = [
        "How do I authenticate with the API?",
        "What happens when I exceed the rate limit?",
        "How long do access tokens last?",
        "What error code do I get for invalid parameters?"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        print("-" * 40)
        answer = rag.query(query, k=2)
        print(f"A: {answer}\n")

def demo_product_support():
    """Demo: Product support knowledge base"""
    print("\n" + "="*60)
    print("DEMO 2: Product Support Knowledge Base")
    print("="*60)
    
    rag = ClaudeRAG()
    
    # Sample product support documentation
    support_docs = [
        """
        Account Setup and Login Issues
        
        Common login problems and solutions:
        
        Forgot Password:
        1. Click "Forgot Password" on login page
        2. Enter your registered email address
        3. Check email for reset link (check spam folder)
        4. Click link and create new password
        
        Account Locked:
        - After 5 failed login attempts, account is locked for 15 minutes
        - Contact support if you need immediate access
        
        Two-Factor Authentication Issues:
        - Ensure your device time is synchronized
        - Try generating a new backup code
        - Contact support to disable 2FA temporarily
        """,
        """
        Billing and Subscription Management
        
        Subscription Plans:
        - Basic: $9.99/month - includes core features
        - Pro: $19.99/month - includes advanced analytics  
        - Enterprise: $49.99/month - includes API access and priority support
        
        Payment Issues:
        - We accept all major credit cards and PayPal
        - Billing cycle starts from subscription date
        - Failed payments result in 7-day grace period
        - After grace period, account is downgraded to free tier
        
        Cancellation:
        - Can cancel anytime from account settings
        - No refunds for partial months
        - Account remains active until end of billing period
        """,
        """
        Feature Troubleshooting
        
        Data Import Issues:
        - Supported formats: CSV, JSON, XML
        - Maximum file size: 100MB
        - Ensure UTF-8 encoding for international characters
        - Check for missing required columns
        
        Performance Problems:
        - Large datasets may take 5-10 minutes to process
        - Try smaller batches if import fails
        - Clear browser cache if interface is slow
        - Use Chrome or Firefox for best performance
        
        Export Functionality:
        - Available in Pro and Enterprise plans
        - Exports limited to 50,000 records per request
        - Multiple export formats: PDF, Excel, CSV
        """
    ]
    
    sources = ["login_support", "billing_support", "feature_support"]
    rag.add_documents(support_docs, sources)
    
    queries = [
        "I forgot my password, how do I reset it?",
        "What payment methods do you accept?",
        "My data import is failing, what should I check?",
        "How do I cancel my subscription?"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        print("-" * 40)
        answer = rag.query(query, k=2)
        print(f"A: {answer}\n")

def demo_research_papers():
    """Demo: Research paper analysis"""
    print("\n" + "="*60)
    print("DEMO 3: Research Paper Analysis")
    print("="*60)
    
    rag = ClaudeRAG()
    
    # Sample research abstracts and key findings
    research_docs = [
        """
        Title: "Attention Is All You Need" - Transformer Architecture
        
        Abstract: We propose a new simple network architecture, the Transformer, 
        based solely on attention mechanisms, dispensing with recurrence and 
        convolutions entirely. Experiments on two machine translation tasks show 
        these models to be superior in quality while being more parallelizable 
        and requiring significantly less time to train.
        
        Key Findings:
        - Multi-head attention allows model to jointly attend to information 
          from different representation subspaces
        - Positional encoding enables the model to make use of sequence order
        - Transformer achieves 28.4 BLEU on WMT 2014 English-German translation
        - Training time reduced from 3.5 days to 12 hours on 8 P100 GPUs
        """,
        """
        Title: "BERT: Pre-training of Deep Bidirectional Transformers"
        
        Abstract: We introduce BERT, which stands for Bidirectional Encoder 
        Representations from Transformers. BERT is designed to pre-train deep 
        bidirectional representations from unlabeled text by jointly conditioning 
        on both left and right context in all layers.
        
        Key Contributions:
        - Bidirectional training allows deeper understanding of language context
        - Masked Language Model (MLM) objective enables bidirectional training
        - Next Sentence Prediction (NSP) helps with sentence relationships
        - Achieves state-of-the-art results on 11 NLP tasks
        - BERT-Large: 24 layers, 1024 hidden size, 340M parameters
        """,
        """
        Title: "Language Models are Few-Shot Learners" - GPT-3
        
        Abstract: We train GPT-3, an autoregressive language model with 175 
        billion parameters, and test its performance in the few-shot setting. 
        GPT-3 achieves strong performance on many NLP datasets, including 
        translation, question-answering, and cloze tasks.
        
        Key Results:
        - 175 billion parameters trained on 570GB of text
        - Strong few-shot performance without task-specific fine-tuning
        - Emergent abilities: arithmetic, word unscrambling, novel word usage
        - In-context learning: performance improves with more examples
        - Scaling laws: performance predictably improves with model size
        """
    ]
    
    sources = ["transformer_paper", "bert_paper", "gpt3_paper"]
    rag.add_documents(research_docs, sources)
    
    queries = [
        "What are the key innovations in the Transformer architecture?",
        "How does BERT's training differ from previous language models?",
        "What capabilities emerge in large language models like GPT-3?",
        "Compare the training approaches of BERT vs GPT-3"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        print("-" * 40)
        answer = rag.query(query, k=3)
        print(f"A: {answer}\n")

def demo_custom_parameters():
    """Demo: Custom parameter tuning"""
    print("\n" + "="*60)
    print("DEMO 4: Custom Parameter Tuning")
    print("="*60)
    
    rag = ClaudeRAG()
    
    # Add some sample content
    docs = [
        "The quick brown fox jumps over the lazy dog. " * 50,  # Repeated content
        "Machine learning is a subset of artificial intelligence. " * 50,
        "Python is a popular programming language for data science. " * 50
    ]
    
    rag.add_documents(docs, ["doc1", "doc2", "doc3"])
    
    query = "What is machine learning?"
    
    print("Testing different retrieval parameters:\n")
    
    # Test different k values
    for k in [1, 3, 5]:
        print(f"Retrieving top {k} chunks:")
        answer = rag.query(query, k=k, max_context_length=2000)
        print(f"Answer length: {len(answer)} characters")
        print(f"Answer: {answer[:100]}...\n")
    
    # Test different context lengths
    print("Testing different context lengths:\n")
    for max_len in [1000, 3000, 5000]:
        print(f"Max context length: {max_len} tokens")
        answer = rag.query(query, k=3, max_context_length=max_len)
        print(f"Answer: {answer[:100]}...\n")

def main():
    """Run all demonstrations"""
    print("Claude RAG System - Use Case Demonstrations")
    print("This will demonstrate various applications of RAG with Claude API")
    
    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("\nError: Please set your API keys in .env file")
        print("Copy env_example.txt to .env and add your keys")
        return
    
    try:
        demo_tech_documentation()
        demo_product_support()
        demo_research_papers()
        demo_custom_parameters()
        
        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    main() 