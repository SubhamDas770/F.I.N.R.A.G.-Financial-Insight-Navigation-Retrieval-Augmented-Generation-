import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    metadata: Dict[str, Any]
    child_ids: List[str] = field(default_factory=list)


class ParentChildChunker:
    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 50,
    ):
        """
        Initializes splitters for parent and child chunks.
        Separators preserve structural markdown elements like tables and headers.
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

        # Custom separators ensure table blocks and headers aren't broken prematurely
        separators = ["\n[TABLE_END]\n", "\n\n", "\n", " ", ""]

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=separators,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=separators,
        )

    def process_pages(
        self, parsed_pages: List[Dict[str, Any]]
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Takes page dicts from DocumentParser and generates hierarchical chunks.
        
        Returns:
            Tuple containing list of ParentChunks and list of ChildChunks.
        """
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []

        for page in parsed_pages:
            page_text = page["content"]
            page_num = page["page_number"]
            source = page["source"]

            if not page_text.strip():
                continue

            # Step 1: Split page content into parent chunks
            raw_parents = self.parent_splitter.split_text(page_text)

            for parent_index, p_text in enumerate(raw_parents):
                parent_id = str(uuid.uuid4())
                parent_metadata = {
                    "source": source,
                    "page_number": page_num,
                    "parent_index": parent_index,
                    "chunk_type": "parent",
                }

                # Step 2: Split parent chunk into smaller child chunks
                raw_children = self.child_splitter.split_text(p_text)
                current_child_ids = []

                for child_index, c_text in enumerate(raw_children):
                    child_id = str(uuid.uuid4())
                    current_child_ids.append(child_id)

                    child_metadata = {
                        "source": source,
                        "page_number": page_num,
                        "parent_id": parent_id,
                        "child_index": child_index,
                        "chunk_type": "child",
                    }

                    child_chunks.append(
                        ChildChunk(
                            child_id=child_id,
                            parent_id=parent_id,
                            text=c_text,
                            metadata=child_metadata,
                        )
                    )

                # Store Parent Chunk with references to child IDs
                parent_chunks.append(
                    ParentChunk(
                        parent_id=parent_id,
                        text=p_text,
                        metadata=parent_metadata,
                        child_ids=current_child_ids,
                    )
                )

        print(
            f"✂️ Chunking Complete: Created {len(parent_chunks)} Parent Chunks "
            f"and {len(child_chunks)} Child Chunks from {len(parsed_pages)} pages."
        )
        return parent_chunks, child_chunks


if __name__ == "__main__":
    # Smoke test for chunker
    mock_pages = [
        {
            "page_number": 1,
            "source": "quarterly_report.pdf",
            "content": (
                "The enterprise software division achieved $12.5M ARR in Q3. "
                "Migration to serverless infrastructure reduced operational overhead by 34%.\n\n"
                "[TABLE_START]\n| Metric | Value |\n|---|---|\n| ARR | $12.5M |\n"
                "| Churn Rate | 1.2% |\n[TABLE_END]\n\n"
                "Customer acquisition cost decreased significantly across enterprise segments."
            ),
        }
    ]

    chunker = ParentChildChunker(
        parent_chunk_size=400,
        parent_chunk_overlap=50,
        child_chunk_size=100,
        child_chunk_overlap=20,
    )

    parents, children = chunker.process_pages(mock_pages)

    if parents and children:
        print("\n--- Parent Sample ---")
        print(f"ID: {parents[0].parent_id}")
        print(f"Text: {parents[0].text}")
        print(f"Child IDs: {parents[0].child_ids}")

        print("\n--- Child Sample ---")
        print(f"ID: {children[0].child_id}")
        print(f"Parent Ref: {children[0].parent_id}")
        print(f"Text: {children[0].text}")