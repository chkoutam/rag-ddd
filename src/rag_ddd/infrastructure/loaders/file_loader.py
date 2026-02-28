from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from pypdf import PdfReader

from rag_ddd.domain.entities import Document
from rag_ddd.domain.ports import DocumentLoader


class FileLoader(DocumentLoader):
    def __init__(
        self,
        allowed_extensions: set[str] | None = None,
        progress: bool = False,
    ) -> None:
        self._allowed_extensions = {ext.lower() for ext in allowed_extensions} if allowed_extensions else None
        self._progress = progress

    def load(self, path: str) -> List[Document]:
        target = Path(path)
        files = [target] if target.is_file() else list(target.rglob("*"))
        documents: List[Document] = []

        for file_path in files:
            if not file_path.is_file():
                continue
            if self._allowed_extensions is not None and file_path.suffix.lower() not in self._allowed_extensions:
                continue
            if self._progress:
                print(f"Loading: {file_path}")
            if file_path.suffix.lower() in {".txt", ".md"}:
                text = file_path.read_text(encoding="utf-8")
            elif file_path.suffix.lower() == ".pdf":
                text = self._read_pdf(file_path)
            else:
                continue
            if not text.strip():
                if self._progress:
                    print(f"Skipped (empty): {file_path}")
                continue
            documents.append(
                Document(
                    doc_id=uuid.uuid4().hex,
                    text=text,
                    metadata={"source": str(file_path)},
                )
            )
            if self._progress:
                print(f"Loaded: {file_path}")
        return documents

    def _read_pdf(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
        except Exception:
            return ""

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        # Remove Unicode surrogates that pypdf sometimes produces
        return text.encode("utf-8", errors="replace").decode("utf-8")
