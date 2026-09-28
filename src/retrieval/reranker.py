"""BGE Cross-Encoder Reranker implementation.

Re-scores and re-ranks retrieved candidates using BAAI/bge-reranker-large
to maximize contextual precision before feeding to LLM.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from src.config import settings


class BGEReranker:
    """Reranks candidate documents using BGE Cross-Encoder."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load model to avoid memory overhead if not requested immediately."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = settings.rerank_top_n,
    ) -> List[Dict[str, Any]]:
        """Reranks documents by computing cross-encoder scores against the query.

        Args:
            query: The user query string.
            documents: Candidate dicts from vector search (each containing 'parent_content' or 'child_content').
            top_n: How many top-ranked documents to return.

        Returns:
            Top-N reranked document dictionaries with added 'rerank_score'.
        """
        if not documents:
            return []

        # Form (query, document_text) pairs - using parent_content if available, else child_content
        pairs = [
            (query, doc.get("parent_content") or doc.get("child_content", ""))
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        for idx, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[idx])

        # Sort descending by rerank score
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
