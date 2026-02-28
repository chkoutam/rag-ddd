from __future__ import annotations

import uuid
from typing import Iterable, List

import numpy as np
from rag_ddd.domain.entities import Chunk, Document
from rag_ddd.domain.ports import Chunker


class SemanticChunker(Chunker):
    """Splits text at semantic boundaries using sentence embeddings.

    Instead of cutting at a fixed token count, this chunker:
    1. Splits text into sentences
    2. Embeds each sentence
    3. Finds breakpoints where semantic similarity drops
    4. Groups consecutive similar sentences into chunks
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ) -> None:
        self._model_name = embedding_model_name
        self._similarity_threshold = similarity_threshold
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def chunk(self, documents: Iterable[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        return chunks

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        sentences = self._split_sentences(doc.text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    chunk_id=uuid.uuid4().hex,
                    doc_id=doc.doc_id,
                    text=sentences[0],
                    metadata=doc.metadata,
                )
            ]

        model = self._get_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self._similarity_threshold:
                breakpoints.append(i + 1)

        # Group sentences into chunks based on breakpoints
        groups: List[List[str]] = []
        start = 0
        for bp in breakpoints:
            group = sentences[start:bp]
            groups.append(group)
            start = bp
        groups.append(sentences[start:])

        # Merge small groups, split large groups
        chunks: List[Chunk] = []
        buffer: List[str] = []

        for group in groups:
            buffer.extend(group)
            buffer_text = " ".join(buffer)

            if len(buffer_text) >= self._min_chunk_size:
                # Split if too large
                while len(buffer_text) > self._max_chunk_size:
                    cut = self._find_sentence_cut(buffer, self._max_chunk_size)
                    chunk_text = " ".join(buffer[:cut])
                    chunks.append(
                        Chunk(
                            chunk_id=uuid.uuid4().hex,
                            doc_id=doc.doc_id,
                            text=chunk_text.strip(),
                            metadata=doc.metadata,
                        )
                    )
                    buffer = buffer[cut:]
                    buffer_text = " ".join(buffer)

                if buffer:
                    chunks.append(
                        Chunk(
                            chunk_id=uuid.uuid4().hex,
                            doc_id=doc.doc_id,
                            text=buffer_text.strip(),
                            metadata=doc.metadata,
                        )
                    )
                    buffer = []

        # Flush remaining buffer
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=uuid.uuid4().hex,
                        doc_id=doc.doc_id,
                        text=text,
                        metadata=doc.metadata,
                    )
                )

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        import re

        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _find_sentence_cut(sentences: List[str], max_size: int) -> int:
        total = 0
        for i, s in enumerate(sentences):
            total += len(s) + 1  # +1 for space
            if total > max_size:
                return max(1, i)
        return len(sentences)
