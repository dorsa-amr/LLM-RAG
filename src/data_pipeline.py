"""
Data pipeline for fetching and processing PubMed articles.
"""

from typing import List, Dict
import requests
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings


class PubMedFetcher:
    """Fetches articles from PubMed using the E-Utilities API."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

    @staticmethod
    def _normalize_pmcid(value: str) -> str:
        """Normalize PMCID to canonical format (e.g., PMC12345)."""
        raw = str(value or "").strip()
        if not raw:
            return ""
        raw = raw.upper()
        if not raw.startswith("PMC"):
            raw = f"PMC{raw}"
        return raw.split(".", 1)[0]

    @staticmethod
    def _get_with_retries(url: str, params: Dict[str, str], timeout: int = 15, retries: int = 3) -> requests.Response:
        """HTTP GET with light retry/backoff for transient API limits and network issues."""
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": "LLM-RAG/1.0 (mailto:user@example.com)"},
                )
                response.raise_for_status()
                return response
            except Exception as e:
                last_error = e
                # Backoff for rate limits/transient issues.
                sleep_s = 0.75 * (attempt + 1)
                if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429:
                    sleep_s = 2 ** (attempt + 1)
                time.sleep(sleep_s)
        raise last_error
    
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
        pmid_to_pmcid = {}
        abstracts_by_pmid = PubMedFetcher._fetch_pubmed_abstracts(pmids)

        if settings.pubmed_use_full_text and pmids:
            pmid_to_pmcid = PubMedFetcher._fetch_pmcid_map(pmids)

        pmc_full_texts = {}
        if pmid_to_pmcid:
            pmcids = [pmcid for pmcid in pmid_to_pmcid.values() if pmcid]
            pmc_full_texts = PubMedFetcher._fetch_pmc_full_texts(pmcids)
        
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
                response = PubMedFetcher._get_with_retries(
                    f"{PubMedFetcher.BASE_URL}/esummary.fcgi",
                    params=params,
                    timeout=15,
                )
                
                data = response.json()
                result = data.get("result", {})
                uids = result.get("uids", [])
                
                for uid in uids:
                    article_data = result.get(uid, {})
                    # Keep article if it has a title or text content.
                    if article_data.get("title") or article_data.get("abstract"):
                        pmcid = PubMedFetcher._normalize_pmcid(pmid_to_pmcid.get(uid, ""))
                        full_text = pmc_full_texts.get(pmcid, "")
                        articles.append({
                            "pmid": uid,
                            "pmcid": pmcid,
                            "title": article_data.get("title", ""),
                            "abstract": article_data.get("abstract", "") or abstracts_by_pmid.get(uid, ""),
                            "full_text": full_text,
                            "authors": article_data.get("authors", []),
                            "pub_date": article_data.get("pubdate", "")
                        })
            except Exception as e:
                print(f"Error fetching batch: {e}")
                continue
        
        return articles

    @staticmethod
    def _fetch_pmcid_map(pmids: List[str]) -> Dict[str, str]:
        """Map PMIDs to PMCIDs where full text is available in PMC."""
        mapping: Dict[str, str] = {}
        batch_size = 50

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            params = {
                "ids": ",".join(batch),
                "format": "json",
                "tool": "rag-system",
                "email": "user@example.com",
            }
            try:
                response = PubMedFetcher._get_with_retries(
                    PubMedFetcher.PMC_IDCONV_URL,
                    params=params,
                    timeout=20,
                )
                data = response.json()
                for record in data.get("records", []):
                    pmid = str(record.get("pmid", "")).strip()
                    pmcid = PubMedFetcher._normalize_pmcid(record.get("pmcid", ""))
                    if pmid and pmcid:
                        mapping[pmid] = pmcid
            except Exception as e:
                print(f"Warning: PMC ID mapping failed for a batch: {e}")

            # Be polite to the service and reduce probability of 429 responses.
            time.sleep(0.34)

        return mapping

    @staticmethod
    def _fetch_pubmed_abstracts(pmids: List[str]) -> Dict[str, str]:
        """Fetch abstracts from PubMed efetch XML by PMID."""
        abstracts: Dict[str, str] = {}
        if not pmids:
            return abstracts

        batch_size = settings.pubmed_batch_size
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "tool": "rag-system",
                "email": "user@example.com",
            }

            try:
                response = PubMedFetcher._get_with_retries(
                    f"{PubMedFetcher.BASE_URL}/efetch.fcgi",
                    params=params,
                    timeout=20,
                )
                root = ET.fromstring(response.text)
                for article in root.findall(".//PubmedArticle"):
                    pmid_node = article.find(".//PMID")
                    if pmid_node is None or not pmid_node.text:
                        continue
                    pmid = pmid_node.text.strip()

                    abstract_parts = []
                    for abs_node in article.findall(".//Abstract/AbstractText"):
                        label = abs_node.attrib.get("Label", "").strip()
                        text = " ".join(" ".join(abs_node.itertext()).split())
                        if text:
                            abstract_parts.append(f"{label}: {text}" if label else text)

                    if abstract_parts:
                        abstracts[pmid] = "\n".join(abstract_parts)
            except Exception as e:
                print(f"Warning: failed to fetch PubMed abstracts for a batch: {e}")

        return abstracts

    @staticmethod
    def _fetch_pmc_full_texts(pmcids: List[str]) -> Dict[str, str]:
        """Fetch full text for PMCIDs and return cleaned text by PMCID."""
        full_texts: Dict[str, str] = {}
        if not pmcids:
            return full_texts

        batch_size = 20
        for i in tqdm(range(0, len(pmcids), batch_size), desc="Fetching PMC full text"):
            batch = pmcids[i:i + batch_size]
            params = {
                "db": "pmc",
                "id": ",".join(batch),
                "retmode": "xml",
                "tool": "rag-system",
                "email": "user@example.com",
            }

            try:
                response = PubMedFetcher._get_with_retries(
                    f"{PubMedFetcher.BASE_URL}/efetch.fcgi",
                    params=params,
                    timeout=30,
                )
                batch_texts = PubMedFetcher._parse_pmc_full_text_xml(response.text)
                full_texts.update(batch_texts)
            except Exception as e:
                print(f"Warning: failed to fetch PMC full text batch: {e}")

        return full_texts

    @staticmethod
    def _parse_pmc_full_text_xml(xml_text: str) -> Dict[str, str]:
        """Parse PMC efetch XML and return body text by PMCID."""
        parsed: Dict[str, str] = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return parsed

        def local_name(tag: str) -> str:
            return tag.split("}", 1)[1] if "}" in tag else tag

        for article in [el for el in root.iter() if local_name(el.tag) == "article"]:
            pmcid = ""
            for article_id in [el for el in article.iter() if local_name(el.tag) == "article-id"]:
                pub_id_type = (article_id.attrib.get("pub-id-type") or "").lower()
                if pub_id_type in {"pmc", "pmcid"} and article_id.text:
                    id_text = article_id.text.strip()
                    pmcid = PubMedFetcher._normalize_pmcid(id_text)
                    break

            body = next((el for el in article.iter() if local_name(el.tag) == "body"), None)
            if not pmcid or body is None:
                continue

            full_text = " ".join(body.itertext())
            full_text = " ".join(full_text.split())
            if not full_text:
                continue

            if settings.pubmed_full_text_char_limit > 0:
                full_text = full_text[:settings.pubmed_full_text_char_limit]

            parsed[pmcid] = full_text

        return parsed


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
        full_text_count = 0
        abstract_count = 0
        
        for article in tqdm(articles, desc="Processing articles"):
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            full_text = article.get("full_text", "")

            if full_text:
                text = f"Title: {title}\n\nFull Text: {full_text}"
                source_type = "PMC full text"
                full_text_count += 1
            elif abstract:
                text = f"Title: {title}\n\nAbstract: {abstract}"
                source_type = "PubMed abstract"
                abstract_count += 1
            else:
                # If both abstract and full text are missing, skip the record.
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
                        "pmcid": article.get("pmcid", ""),
                        "chunk_idx": i,
                        "source": "PubMed",
                        "source_type": source_type,
                    }
                )
                documents.append(doc)

        print(f"Used {full_text_count} full-text articles and {abstract_count} abstract-only articles")
        
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
