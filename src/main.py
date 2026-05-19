"""
Main entry point for the RAG system.
"""

import sys
import os

# Allow running this script from either project root or the src folder.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline import pipeline
from src.retriever import get_retriever
from src.agent import create_qa_agent, query_agent


def main():
    """Main application flow."""
    
    print("\n" + "="*70)
    print("AGENTIC RAG SYSTEM FOR PUBMED")
    print("="*70 + "\n")
    
    # Check if vector DB is populated
    retriever = get_retriever()
    
    # Step 1: Load sample data (optional)
    setup_data = input("Would you like to load sample PubMed articles? (y/n): ").lower() == 'y'
    
    if setup_data:
        query = input("Enter search query for PubMed (default: 'machine learning'): ").strip()
        if not query:
            query = "machine learning"
        
        # Fetch and process data
        documents = pipeline(query, max_articles=50)
        
        if documents:
            # Add to vector store
            retriever.vector_store.add_documents(documents)
            print(f"\n✓ Loaded {len(documents)} document chunks into vector database\n")
        else:
            print("No documents to load.\n")
    
    # Step 2: Create agent
    print("Initializing QA agent...")
    agent = create_qa_agent(memory_enabled=True)
    print("✓ Agent ready\n")
    
    # Step 3: Interactive QA loop
    print("="*70)
    print("INTERACTIVE QA SESSION")
    print("="*70)
    print("Ask questions about the loaded documents.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\n" + "-"*70)
        print("Searching and reasoning...")
        print("-"*70 + "\n")
        
        response = query_agent(agent, question)
        
        print("\n" + "-"*70)
        print("ANSWER:")
        print("-"*70)
        print(response)
        print("-"*70)


def demo():
    """Run a quick demo without interaction."""
    print("\n" + "="*70)
    print("AGENTIC RAG SYSTEM - DEMO MODE")
    print("="*70 + "\n")
    
    # Load sample questions
    sample_questions = [
        "What are the recent advances in deep learning?",
        "Summarize the findings about neural networks.",
        "What is the relationship between machine learning and AI?"
    ]
    
    print("Creating agent...")
    agent = create_qa_agent(memory_enabled=True)
    print("✓ Agent ready\n")
    
    for question in sample_questions:
        print("\n" + "="*70)
        print(f"Question: {question}")
        print("="*70)
        
        response = query_agent(agent, question)
        print(f"\nAnswer:\n{response}\n")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
