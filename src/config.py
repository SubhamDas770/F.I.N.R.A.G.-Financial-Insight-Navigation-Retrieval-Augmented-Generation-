"""Configuration module for Enterprise RAG.

Loads environment variables, manages path definitions, and defines model / database settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Hugging Face
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # Redis Configuration
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_cache_ttl_seconds: int = Field(default=86400, alias="REDIS_CACHE_TTL_SECONDS")  # 24 hours

    # Vector DB Settings
    vector_db_path: str = Field(default=str(BASE_DIR / "data" / "vector_db"), alias="VECTOR_DB_PATH")
    collection_name: str = Field(default="enterprise_docs", alias="COLLECTION_NAME")

    # Model Selection
    base_model_name: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct", alias="BASE_MODEL_NAME")
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL_NAME")
    reranker_model_name: str = Field(default="BAAI/bge-reranker-large", alias="RERANKER_MODEL_NAME")

    # Paths
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    fine_tuned_model_path: str = Field(default=str(BASE_DIR / "data" / "fine_tuned_llama3"), alias="FINE_TUNED_MODEL_PATH")

    # Chunking Parameters (Parent-Child Strategy)
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # Retrieval & Reranking Defaults
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    semantic_cache_threshold: float = 0.92

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
