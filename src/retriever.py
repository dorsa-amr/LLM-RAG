"""
Vector database and retrieval pipeline for semantic search over documents.
"""

import os
from typing import List, Dict, Any
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config import settings


class VectorStore:
    """Manages vector database operations using Chroma."""
    
    def __init__(self):
        """Initialize vector store with Chroma backend."""
        # Create directories if they don't exist
        os.makedirs(settings.chroma_db_path, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            embedding_function=self.embeddings,
            collection_name="pubmed_articles"
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        print("Documents added successfully")
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Semantic search over documents.
        
        Args:
            query: Search query
            k: Number of results to return (default: settings.retrieval_top_k)
            
        Returns:
            List of relevant documents
        """
        if k is None:
            k = settings.retrieval_top_k
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """
        Semantic search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if k is None:
            k = settings.retrieval_top_k
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def delete_collection(self) -> None:
        """Delete the collection (useful for resetting)."""
        self.client.delete_collection("pubmed_articles")
        print("Collection deleted")


class RetrieverPipeline:
    """High-level retrieval pipeline with tunable parameters."""
    
    def __init__(self):
        """Initialize the retriever."""
        self.vector_store = VectorStore()
    
    def retrieve(self, query: str, top_k: int = None, score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of results (uses config default if None)
            score_threshold: Minimum similarity threshold (uses config if None)
            
        Returns:
            List of dicts with document content and metadata
        """
        if top_k is None:
            top_k = settings.retrieval_top_k
        if score_threshold is None:
            score_threshold = settings.retrieval_score_threshold
        
        results = self.vector_store.search_with_scores(query, k=top_k)
        
        # Chroma similarity_search_with_score returns distance (lower is better)
        filtered_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
            if score <= score_threshold
        ]

        # If strict threshold removes all results, fall back to top matches.
        if not filtered_results and results:
            filtered_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results[:top_k]
            ]
        
        return filtered_results


# Global retriever instance
_retriever = None

def get_retriever() -> RetrieverPipeline:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RetrieverPipeline()
    return _retriever
