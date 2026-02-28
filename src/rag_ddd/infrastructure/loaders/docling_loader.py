from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from rag_ddd.domain.entities import Document
from rag_ddd.domain.ports import DocumentLoader


class DoclingLoader(DocumentLoader):
    """Advanced document loader using Docling (IBM).

    Provides layout-aware parsing for PDFs:
    - Table structure preservation
    - Heading hierarchy detection
    - Reading order correction
    - Image extraction
    - Formula detection
    """

    def __init__(self, progress: bool = False) -> None:
        self._progress = progress
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def load(self, path: str) -> List[Document]:
        target = Path(path)
        files = [target] if target.is_file() else list(target.rglob("*"))
        documents: List[Document] = []

        supported = {".pdf", ".docx", ".pptx", ".html", ".md", ".txt"}

        for file_path in files:
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported:
                continue
            if self._progress:
                print(f"Docling parsing: {file_path}")

            text = self._parse_file(file_path)
            if not text.strip():
                if self._progress:
                    print(f"Skipped (empty): {file_path}")
                continue

            documents.append(
                Document(
                    doc_id=uuid.uuid4().hex,
                    text=text,
                    metadata={
                        "source": str(file_path),
                        "parser": "docling",
                        "filename": file_path.name,
                    },
                )
            )
            if self._progress:
                print(f"Loaded: {file_path} ({len(text)} chars)")

        return documents

    def _parse_file(self, file_path: Path) -> str:
        # Plain text files don't need Docling
        if file_path.suffix.lower() in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8")

        try:
            converter = self._get_converter()
            result = converter.convert(str(file_path))
            return result.document.export_to_markdown()
        except Exception as exc:
            if self._progress:
                print(f"Docling error on {file_path}: {exc}")
            return ""
