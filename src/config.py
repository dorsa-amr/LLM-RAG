import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """Application configuration from environment variables."""
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 2000))
    
    # Vector DB
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    
    # PubMed
    pubmed_batch_size: int = int(os.getenv("PUBMED_BATCH_SIZE", 100))
    pubmed_max_articles: int = int(os.getenv("PUBMED_MAX_ARTICLES", 500))
    
    # Retrieval
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", 5))
    retrieval_score_threshold: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.5))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
