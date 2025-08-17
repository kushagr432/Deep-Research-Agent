"""
Configuration management for Financial Research Chatbot
"""
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # FastAPI Configuration
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    fastapi_reload: bool = True
    fastapi_log_level: str = "info"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_client_id: str = "financial_chatbot"
    kafka_group_id: str = "financial_chatbot_group"
    
    # Vector Database Configuration
    vector_db_type: str = "mock"  # Options: mock, pinecone, faiss
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "financial_knowledge"
    
    # LLM Configuration (for future use)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    secret_key: str = "your_secret_key_here_change_in_production"
    access_token_expire_minutes: int = 30
    
    # Performance Configuration
    cache_ttl_seconds: int = 3600
    vector_search_top_k: int = 5
    vector_similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings
