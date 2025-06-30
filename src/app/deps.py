# src/app/deps.py
from functools import lru_cache
from .services.rag import RAGService
from .services.recommender import RecommendationService

@lru_cache            
def get_rag() -> RAGService:
    return RAGService()

@lru_cache
def get_rec() -> RecommendationService:
    rag = get_rag()
    return RecommendationService(rag.vectordb)
