# src/app/deps.py
"""
deps
====

Singleton dependency providers for FastAPI routes.

Provides:
- One persistent :class:`RAGService` instance.
- One persistent :class:`RecommendationService` instance sharing
  the same vector store.

Usage
-----
Functions here are injected via ``Depends()`` at runtime.
"""
from functools import lru_cache
from .services.rag import RAGService
from .services.recommender import RecommendationService

@lru_cache            
def get_rag() -> RAGService:
    """Singleton provider for the RAG service instance."""
    return RAGService()

@lru_cache
def get_rec() -> RecommendationService:
    """Singleton provider for the recommendation service instance (shares RAG's vectordb)."""
    rag = get_rag()
    return RecommendationService(rag.vectordb)
