from __future__ import annotations

import uuid
from typing import Iterable, List

import tiktoken

from rag_ddd.domain.entities import Chunk, Document
from rag_ddd.domain.ports import Chunker


class RecursiveChunker(Chunker):
    def __init__(self, chunk_size: int, chunk_overlap: int, progress: bool = False) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._progress = progress

    def chunk(self, documents: Iterable[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            if self._progress:
                source = doc.metadata.get("source", "unknown")
                print(f"Chunking: {source} (chars={len(doc.text)})")
            tokens = self._encoding.encode(doc.text)
            if self._progress:
                print(f"Tokens: {len(tokens)}")
            start = 0
            while start < len(tokens):
                end = min(start + self._chunk_size, len(tokens))
                chunk_text = self._encoding.decode(tokens[start:end])
                chunks.append(
                    Chunk(
                        chunk_id=uuid.uuid4().hex,
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        metadata=doc.metadata,
                    )
                )
                if end == len(tokens):
                    break
                start = max(0, end - self._chunk_overlap)
        return chunks
