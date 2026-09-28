import os
import glob
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Configuration
COLLECTION_NAME = "enterprise_docs"
QDRANT_PATH = "./data/vector_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # High-performance 384-dim model for finance

class SECIngestionPipeline:
    def __init__(self, parent_chunk_size: int = 1800, child_chunk_size: int = 400, overlap: int = 50):
        self.parent_size = parent_chunk_size
        self.child_size = child_chunk_size
        self.overlap = overlap
        
        print(f"🧠 Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.vector_dim = self.encoder.get_sentence_embedding_dimension()

        print(f"🗄️ Connecting to local Qdrant Vector DB at {QDRANT_PATH}...")
        self.qdrant = QdrantClient(path=QDRANT_PATH)
        self._init_collection()

    def _init_collection(self):
        """Creates collection if it doesn't already exist."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            print(f"✨ Created Qdrant collection: {COLLECTION_NAME}")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from PDF with metadata."""
        reader = PdfReader(pdf_path)
        doc_name = os.path.basename(pdf_path)
        pages_content = []

        print(f"📄 Processing {doc_name} ({len(reader.pages)} pages)...")
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_content.append({
                    "text": text.strip(),
                    "page_number": i + 1,
                    "doc_name": doc_name
                })
        return pages_content

    def _split_into_chunks(self, text: str, size: int, overlap: int) -> List[str]:
        """Sliding window chunker."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    def process_and_index(self, pdf_path: str):
        pages = self.extract_text_from_pdf(pdf_path)
        points = []
        point_id = 0

        print("⚡ Performing Parent-Child Chunking & Vectorization...")
        for page in pages:
            # Step 1: Create Parent Chunks
            parents = self._split_into_chunks(page["text"], self.parent_size, self.overlap)
            
            for p_idx, parent_text in enumerate(parents):
                # Step 2: Create Child Chunks inside each Parent
                children = self._split_into_chunks(parent_text, self.child_size, self.overlap)
                
                for c_idx, child_text in enumerate(children):
                    # Step 3: Embed only the Child Chunk
                    vector = self.encoder.encode(child_text).tolist()
                    
                    # Step 4: Construct Point with Child vector + Parent full context in payload
                    payload = {
                        "doc_name": page["doc_name"],
                        "page_number": page["page_number"],
                        "parent_id": f"{page['page_number']}_{p_idx}",
                        "child_text": child_text,
                        "parent_context": parent_text  # Passed to LLM for rich answer generation
                    }
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    ))
                    point_id += 1

        # Upsert in batches of 100
        batch_size = 100
        print(f"📦 Upserting {len(points)} child vectors into Qdrant...")
        for i in range(0, len(points), batch_size):
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + batch_size]
            )

        print(f"✅ Ingestion complete for {os.path.basename(pdf_path)}!")

if __name__ == "__main__":
    pipeline = SECIngestionPipeline()
    
    # Ingest all PDF files found in ./data/
    pdf_files = glob.glob("./data/*.pdf")
    if not pdf_files:
        print("⚠️ No PDF files found in ./data/. Please place your 10-K PDF in ./data/ directory.")
    else:
        for pdf in pdf_files:
            pipeline.process_and_index(pdf)