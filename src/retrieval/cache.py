import hashlib
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
import redis
from sentence_transformers import SentenceTransformer


class RedisSemanticCache:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.92,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        ttl_seconds: int = 86400,  # 24 hours default TTL
    ):
        """
        Dual-layer cache:
        1. Exact Match: SHA256(query) -> Cached Response
        2. Semantic Match: Query Vector Cosine Similarity -> Cached Response
        """
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl_seconds
        
        # Initialize embedding model for semantic query comparison
        print(f"🔄 Cache Encoder Loading: {embedding_model_name}...")
        self.encoder = SentenceTransformer(embedding_model_name)

        # Connect to Redis
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
        )
        try:
            self.redis.ping()
            self.enabled = True
            print("✅ Redis Semantic Cache connected.")
        except redis.ConnectionError:
            print("⚠️ Redis unavailable. Cache running in in-memory fallback mode.")
            self.enabled = False
            self.memory_fallback: Dict[str, Dict[str, Any]] = {}

    def _hash_query(self, query: str) -> str:
        """Generates a normalized SHA256 key for exact match lookup."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
        """
        Checks the cache for a match.
        
        Returns:
            Tuple: (cached_response, metadata, match_type)
            match_type can be 'EXACT', 'SEMANTIC', or 'MISS'
        """
        exact_key = f"cache:exact:{self._hash_query(query)}"

        # -------------------------------------------------------------
        # 1. Exact Match Check O(1)
        # -------------------------------------------------------------
        if self.enabled:
            raw_data = self.redis.get(exact_key)
        else:
            raw_data = json.dumps(self.memory_fallback[exact_key]) if exact_key in self.memory_fallback else None

        if raw_data:
            entry = json.loads(raw_data)
            print(f"⚡ [Cache HIT] Exact match found for query: '{query[:30]}...'")
            return entry["response"], entry["metadata"], "EXACT"

        # -------------------------------------------------------------
        # 2. Semantic Similarity Match Check
        # -------------------------------------------------------------
        query_vector = self.encoder.encode(query, normalize_embeddings=True)

        if self.enabled:
            # Fetch all stored semantic query keys
            semantic_keys = self.redis.keys("cache:semantic:*")
            best_score = -1.0
            best_entry = None

            for key in semantic_keys:
                raw_item = self.redis.get(key)
                if not raw_item:
                    continue
                item = json.loads(raw_item)
                
                cached_vector = np.array(item["vector"], dtype=np.float32)
                # Compute Cosine Similarity (vectors are normalized)
                score = float(np.dot(query_vector, cached_vector))

                if score > best_score:
                    best_score = score
                    best_entry = item

            if best_entry and best_score >= self.similarity_threshold:
                print(f"🎯 [Cache HIT] Semantic match ({best_score:.4f} >= {self.similarity_threshold}): '{query[:30]}...'")
                return best_entry["response"], best_entry["metadata"], "SEMANTIC"

        else:
            # In-memory fallback semantic scan
            best_score = -1.0
            best_entry = None
            for key, item in self.memory_fallback.items():
                if not key.startswith("cache:semantic:"):
                    continue
                cached_vector = np.array(item["vector"], dtype=np.float32)
                score = float(np.dot(query_vector, cached_vector))
                if score > best_score:
                    best_score = score
                    best_entry = item

            if best_entry and best_score >= self.similarity_threshold:
                return best_entry["response"], best_entry["metadata"], "SEMANTIC"

        return None, None, "MISS"

    def set(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Stores an LLM response under both exact and semantic cache keys.
        """
        metadata = metadata or {}
        exact_key = f"cache:exact:{self._hash_query(query)}"
        semantic_key = f"cache:semantic:{self._hash_query(query)}"

        # Embed query vector for semantic search
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

        payload = {
            "query": query,
            "response": response,
            "metadata": metadata,
            "vector": query_vector,
        }

        serialized = json.dumps(payload)

        if self.enabled:
            # Set exact match key
            self.redis.set(exact_key, json.dumps({"response": response, "metadata": metadata}), ex=self.ttl)
            # Set semantic match entry
            self.redis.set(semantic_key, serialized, ex=self.ttl)
        else:
            self.memory_fallback[exact_key] = {"response": response, "metadata": metadata}
            self.memory_fallback[semantic_key] = payload

        print(f"💾 [Cache SET] Saved query response (TTL: {self.ttl}s)")


if __name__ == "__main__":
    # Test Caching Layer
    cache = RedisSemanticCache(similarity_threshold=0.88)

    # 1. Store initial query
    q1 = "What was our revenue in Q3?"
    r1 = "Q3 revenue reached $12.5M."
    meta = {"tokens": 42, "model": "Llama-3-8B-Instruct"}
    
    cache.set(q1, r1, meta)

    # 2. Test exact hit
    resp, m, match_type = cache.get("What was our revenue in Q3?")
    print(f"\nExact Test -> Match Type: {match_type} | Response: '{resp}'")

    # 3. Test semantic hit (rephrased)
    q2 = "How much money did we make in the third quarter?"
    resp_sem, m_sem, match_type_sem = cache.get(q2)
    print(f"\nSemantic Test -> Match Type: {match_type_sem} | Response: '{resp_sem}'")