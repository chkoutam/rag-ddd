"""Extractive summarization for Generative AI documents.

Selects the most important sentences based on position, keywords, and length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


IMPORTANT_KEYWORDS = [
    "model",
    "training",
    "loss",
    "attention",
    "benchmark",
    "performance",
    "architecture",
    "transformer",
    "fine-tuning",
    "pre-training",
    "embedding",
    "generation",
    "inference",
    "accuracy",
    "dataset",
    "evaluation",
    "key",
    "novel",
    "propose",
    "achieve",
    "outperform",
    "state-of-the-art",
    "significantly",
    "demonstrate",
    "result",
    "approach",
    "method",
]


@dataclass
class SummaryResult:
    summary: str
    original_length: int
    summary_length: int

    @property
    def compression_ratio(self) -> float:
        if self.original_length == 0:
            return 0.0
        return self.summary_length / self.original_length


class GenAISummarizer:
    """Extractive summarizer by key sentence selection."""

    def __init__(self, max_sentences: int = 5) -> None:
        self.max_sentences = max_sentences

    def summarize(self, text: str) -> SummaryResult:
        if not text.strip():
            return SummaryResult(summary="", original_length=0, summary_length=0)

        sentences = self._split_sentences(text)

        if len(sentences) <= self.max_sentences:
            summary = " ".join(sentences)
            return SummaryResult(
                summary=summary,
                original_length=len(text),
                summary_length=len(summary),
            )

        scored = [
            (i, sentence, self._score_sentence(sentence, i, len(sentences)))
            for i, sentence in enumerate(sentences)
        ]

        scored.sort(key=lambda x: x[2], reverse=True)
        top_sentences = scored[: self.max_sentences]

        # Restore original order
        top_sentences.sort(key=lambda x: x[0])

        summary = " ".join(s[1] for s in top_sentences)

        return SummaryResult(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        score = 0.0
        sentence_lower = sentence.lower()

        # Position bonus (intro + conclusion)
        if position < 3:
            score += 2.0
        if position >= total - 2:
            score += 1.0

        # Keyword matches
        for keyword in IMPORTANT_KEYWORDS:
            if keyword in sentence_lower:
                score += 1.0

        # Ideal length (50-200 chars)
        length = len(sentence)
        if 50 <= length <= 200:
            score += 1.0
        elif length < 20:
            score -= 1.0

        return score
