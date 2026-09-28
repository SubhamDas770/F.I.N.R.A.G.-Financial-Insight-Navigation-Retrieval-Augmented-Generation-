import json
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


class RAGVectorStore:
    def __init__(
        self,
        collection_name: str = "enterprise_docs",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        qdrant_path: Optional[str] = "./data/vector_db",
    ):
        """
        Initializes the Vector DB (Qdrant) for child embeddings 
        and Redis Key-Value Store for parent text chunks.
        """
        self.collection_name = collection_name
        
        # 1. Initialize Embedding Model
        print(f"🔄 Loading Embedding Model: {embedding_model_name}...")
        self.encoder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.encoder.get_embedding_dimension()

        # 2. Initialize Qdrant Vector DB Client
        if qdrant_path:
            os.makedirs(qdrant_path, exist_ok=True)
            try:
                self.qdrant = QdrantClient(path=qdrant_path)
                print("✅ Qdrant local storage loaded.")
            except RuntimeError as e:
                if "already accessed" in str(e):
                    print("⚠️  Qdrant DB locked by another process (hot-reload). Falling back to in-memory mode.")
                    self.qdrant = QdrantClient(":memory:")
                else:
                    raise
        else:
            self.qdrant = QdrantClient(":memory:")

        self._ensure_qdrant_collection()

        # 3. Initialize Redis Document Store
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
        )
        try:
            self.redis_client.ping()
            print("✅ Connected to Redis Document Store.")
        except redis.ConnectionError:
            print("⚠️ Redis connection failed! Running in mock fallback mode for local testing.")
            self.redis_client = {}

    def _ensure_qdrant_collection(self):
        """Creates the Qdrant vector collection if it does not already exist."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, 
                    distance=Distance.COSINE
                ),
            )
            print(f"📁 Created Qdrant collection: '{self.collection_name}' (dim={self.embedding_dim})")

    def store_parents_and_children(self, parent_chunks: List[Any], child_chunks: List[Any]):
        """
        Ingests Parent Chunks into Redis and Child Chunks into Qdrant.
        """
        # Step A: Store Parent Chunks in Redis (parent_id -> json)
        print(f"📥 Storing {len(parent_chunks)} Parent Chunks in Redis...")
        for parent in parent_chunks:
            key = f"parent:{parent.parent_id}"
            value = json.dumps({
                "parent_id": parent.parent_id,
                "text": parent.text,
                "metadata": parent.metadata,
                "child_ids": parent.child_ids
            })
            if isinstance(self.redis_client, dict):
                self.redis_client[key] = value
            else:
                self.redis_client.set(key, value)

        # Step B: Generate Embeddings for Child Chunks
        print(f"⚡ Embedding and indexing {len(child_chunks)} Child Chunks...")
        child_texts = [c.text for c in child_chunks]
        embeddings = self.encoder.encode(child_texts, show_progress_bar=True, batch_size=32)

        points = []
        for index, (child, vector) in enumerate(zip(child_chunks, embeddings)):
            payload = {
                "child_id": child.child_id,
                "parent_id": child.parent_id,
                "text": child.text,
                **child.metadata
            }
            points.append(
                PointStruct(
                    id=index + 1,
                    vector=vector.tolist(),
                    payload=payload
                )
            )

        # Step C: Upsert Child Points to Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("✅ Ingestion & Vector Indexing Complete.")

    def get_parent_chunk(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a parent chunk from Redis by parent_id."""
        key = f"parent:{parent_id}"
        if isinstance(self.redis_client, dict):
            raw_data = self.redis_client.get(key)
        else:
            raw_data = self.redis_client.get(key)
            
        if raw_data:
            return json.loads(raw_data)
        return None

    def search(self, query: str, top_k_children: int = 10) -> List[Dict[str, Any]]:
        """
        Performs vector search on child chunks, then retrieves corresponding parent chunks.
        
        Returns:
            List of dictionaries containing child match details and full parent context.
        """
        # 1. Embed query
        query_vector = self.encoder.encode(query).tolist()

        # 2. Search Qdrant for top matching child chunks
        search_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k_children,
        ).points

        results = []
        seen_parent_ids = set()

        for hit in search_results:
            child_payload = hit.payload or {}
            parent_id = child_payload.get("parent_id")
            score = hit.score

            # Fetch parent text from Redis if available, or fall back to Qdrant payload
            parent_data = self.get_parent_chunk(parent_id) if parent_id else None
            
            parent_text = None
            if parent_data and isinstance(parent_data, dict):
                parent_text = parent_data.get("text")
            if not parent_text:
                parent_text = child_payload.get("parent_context") or child_payload.get("parent_text")

            child_text = child_payload.get("child_text") or child_payload.get("text") or ""
            if not parent_text:
                parent_text = child_text

            source = child_payload.get("doc_name") or child_payload.get("source") or "Document"
            page_number = child_payload.get("page_number", 1)

            is_new_parent = parent_id not in seen_parent_ids
            if parent_id:
                seen_parent_ids.add(parent_id)

            results.append({
                "score": float(score),
                "child_id": child_payload.get("child_id", str(hit.id)),
                "child_text": child_text,
                "parent_id": parent_id or str(hit.id),
                "parent_text": parent_text,
                "metadata": {
                    "source": source,
                    "page_number": page_number,
                    "is_unique_parent": is_new_parent
                }
            })

        return results


if __name__ == "__main__":
    # Smoke test for VectorStore & Redis Integration
    from src.ingestion.chunker import ParentChildChunker

    # Mock chunk creation
    mock_pages = [{
        "page_number": 1,
        "source": "annual_report.pdf",
        "content": "Cloud infrastructure spending dropped by $45,000 following serverless migration on Modal."
    }]

    chunker = ParentChildChunker(parent_chunk_size=200, child_chunk_size=50)
    parents, children = chunker.process_pages(mock_pages)

    # Initialize store
    store = RAGVectorStore(collection_name="test_collection", qdrant_path=None)
    store.store_parents_and_children(parents, children)

    # Perform query
    query = "How much were hosting costs reduced on Modal?"
    hits = store.search(query, top_k_children=2)

    print("\n🔍 Search Results:")
    for i, res in enumerate(hits, 1):
        print(f"\n--- Result #{i} (Score: {res['score']:.4f}) ---")
        print(f"Child Match: {res['child_text']}")
        print(f"Parent Context: {res['parent_text']}")