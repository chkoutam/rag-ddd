"""Key information extraction for Generative AI documents.

Extracts title, author/instructor, key concepts, and referenced models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from rag_ddd.infrastructure.nlp.ner import GenAINER


@dataclass
class ExtractedInfo:
    title: str | None = None
    author: str | None = None
    key_concepts: List[str] = field(default_factory=list)
    models_mentioned: List[str] = field(default_factory=list)
    papers_cited: List[str] = field(default_factory=list)


AUTHOR_PATTERNS = [
    r"(?:Author|Instructor|Professor|Prof\.?|Dr\.?)\s*:?\s*(.+?)(?:\n|$)",
    r"(?:By|Written by|Presented by)\s+(.+?)(?:\n|,|$)",
]


class GenAIExtractor:
    """Extracts key information from GenAI documents."""

    def __init__(self) -> None:
        self._ner: GenAINER | None = None

    def _get_ner(self) -> GenAINER:
        if self._ner is None:
            self._ner = GenAINER()
        return self._ner

    def extract(self, text: str) -> ExtractedInfo:
        info = ExtractedInfo()

        info.title = self._extract_title(text)
        info.author = self._extract_author(text)

        ner_result = self._get_ner().extract(text)

        info.models_mentioned = list({e.text for e in ner_result.models})
        info.papers_cited = list({e.text for e in ner_result.get_by_label("PAPER_REF")})

        # Combine techniques + models as key concepts
        techniques = {e.text.lower() for e in ner_result.techniques}
        metrics = {e.text for e in ner_result.get_by_label("METRIC")}
        info.key_concepts = sorted(techniques | metrics)

        return info

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for line in text.strip().split("\n")[:10]:
            line = line.strip().lstrip("#").strip()
            if 10 < len(line) < 200:
                return line
        return None

    @staticmethod
    def _extract_author(text: str) -> str | None:
        for pattern in AUTHOR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
