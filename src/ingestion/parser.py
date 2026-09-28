"""PDF Extraction module using PyMuPDF (fitz) and optional LlamaParse support.

Optimized for parsing large enterprise PDFs (100 - 1000+ pages) with page metadata,
cleaning, and structural preservation.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF


class PDFParser:
    """Extracts structured text and metadata from multi-page PDFs."""

    def __init__(self, use_llamaparse: bool = False, llamaparse_api_key: Optional[str] = None):
        self.use_llamaparse = use_llamaparse
        self.llamaparse_api_key = llamaparse_api_key or os.getenv("LLAMA_CLOUD_API_KEY")

    def parse_pdf(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Parses a PDF document into a list of page objects with text and metadata.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of dictionaries containing:
                - 'page_number': int (1-indexed)
                - 'text': str (cleaned extracted text)
                - 'source': str (file name)
                - 'total_pages': int
                - 'metadata': dict (document metadata)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at {path}")

        if self.use_llamaparse and self.llamaparse_api_key:
            return self._parse_with_llamaparse(path)

        return self._parse_with_pymupdf(path)

    def _parse_with_pymupdf(self, path: Path) -> List[Dict[str, Any]]:
        """Extracts text page-by-page using PyMuPDF."""
        pages = []
        doc = fitz.open(str(path))
        total_pages = len(doc)
        doc_metadata = doc.metadata or {}

        for page_idx in range(total_pages):
            page = doc[page_idx]
            # Extract standard text with reading order layout
            text = page.get_text("text").strip()

            if text:  # Skip blank pages
                pages.append({
                    "page_number": page_idx + 1,
                    "text": text,
                    "source": path.name,
                    "file_path": str(path),
                    "total_pages": total_pages,
                    "metadata": {
                        "author": doc_metadata.get("author", ""),
                        "title": doc_metadata.get("title", path.stem),
                        "creation_date": doc_metadata.get("creationDate", ""),
                    }
                })

        doc.close()
        return pages

    def _parse_with_llamaparse(self, path: Path) -> List[Dict[str, Any]]:
        """Fallback / advanced extraction using LlamaParse for complex documents."""
        try:
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError("llama-parse is required when use_llamaparse=True. Run: pip install llama-parse")

        parser = LlamaParse(
            api_key=self.llamaparse_api_key,
            result_type="markdown",
            verbose=True
        )
        documents = parser.load_data(str(path))

        pages = []
        for idx, doc in enumerate(documents):
            pages.append({
                "page_number": idx + 1,
                "text": doc.text.strip(),
                "source": path.name,
                "file_path": str(path),
                "total_pages": len(documents),
                "metadata": getattr(doc, "extra_info", {})
            })
        return pages
