"""
Data pipeline for fetching and processing PubMed articles.
"""

import os
from typing import List, Dict
import requests
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings


class PubMedFetcher:
    """Fetches articles from PubMed using the E-Utilities API."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    @staticmethod
    def search(query: str, max_results: int = None) -> List[str]:
        """
        Search PubMed and get PMIDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of PMIDs
        """
        if max_results is None:
            max_results = settings.pubmed_max_articles
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "tool": "rag-system",
            "email": "user@example.com"
        }
        
        response = requests.get(f"{PubMedFetcher.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()
        
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    
    @staticmethod
    def fetch_articles(pmids: List[str]) -> List[Dict[str, str]]:
        """
        Fetch article details from PMIDs using esummary endpoint.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article data dictionaries
        """
        articles = []
        
        # Fetch in batches using esummary (better for JSON metadata)
        batch_size = settings.pubmed_batch_size
        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching articles"):
            batch = pmids[i:i + batch_size]
            batch_str = ",".join(batch)
            
            params = {
                "db": "pubmed",
                "id": batch_str,
                "retmode": "json",
                "tool": "rag-system",
                "email": "user@example.com"
            }
            
            try:
                response = requests.get(f"{PubMedFetcher.BASE_URL}/esummary.fcgi", params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                result = data.get("result", {})
                uids = result.get("uids", [])
                
                for uid in uids:
                    article_data = result.get(uid, {})
                    # Only add if we have at least a title
                    if article_data.get("title"):
                        articles.append({
                            "pmid": uid,
                            "title": article_data.get("title", ""),
                            "abstract": article_data.get("abstract", ""),
                            "authors": article_data.get("authors", []),
                            "pub_date": article_data.get("pubdate", "")
                        })
            except Exception as e:
                print(f"Error fetching batch: {e}")
                continue
        
        return articles


class DocumentProcessor:
    """Processes raw documents into chunks."""
    
    def __init__(self):
        """Initialize text splitter."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_articles(self, articles: List[Dict[str, str]]) -> List[Document]:
        """
        Convert articles to LangChain Documents with chunks.
        
        Args:
            articles: List of article data dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for article in tqdm(articles, desc="Processing articles"):
            # Combine title and abstract
            text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
            
            # Skip if no content
            if not text.strip():
                continue
            
            # Split into chunks
            chunks = self.splitter.split_text(text)
            
            # Create documents with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "pmid": article["pmid"],
                        "title": article["title"],
                        "pub_date": article["pub_date"],
                        "chunk_idx": i,
                        "source": "PubMed"
                    }
                )
                documents.append(doc)
        
        return documents


def pipeline(query: str = "machine learning", max_articles: int = None) -> List[Document]:
    """
    Complete data pipeline: fetch → process → return documents.
    
    Args:
        query: PubMed search query
        max_articles: Maximum articles to fetch
        
    Returns:
        List of processed documents
    """
    if max_articles is None:
        max_articles = settings.pubmed_max_articles
    
    print(f"\n{'='*60}")
    print(f"DATA PIPELINE: {query}")
    print(f"{'='*60}\n")
    
    # Step 1: Search PubMed
    print(f"Step 1: Searching PubMed for '{query}'...")
    pmids = PubMedFetcher.search(query, max_results=max_articles)
    print(f"Found {len(pmids)} articles\n")
    
    if not pmids:
        print("No articles found!")
        return []
    
    # Step 2: Fetch articles
    print("Step 2: Fetching article details...")
    articles = PubMedFetcher.fetch_articles(pmids)
    print(f"Successfully fetched {len(articles)} articles\n")
    
    # Step 3: Process into documents
    print("Step 3: Processing articles into chunks...")
    processor = DocumentProcessor()
    documents = processor.process_articles(articles)
    print(f"Created {len(documents)} document chunks\n")
    
    return documents


if __name__ == "__main__":
    # Example usage
    docs = pipeline("CRISPR gene editing", max_articles=50)
    print(f"\nTotal documents created: {len(docs)}")
    if docs:
        print(f"\nFirst document:\n{docs[0].page_content[:200]}...")
