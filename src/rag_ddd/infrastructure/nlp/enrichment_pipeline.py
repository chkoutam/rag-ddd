"""NLP Enrichment pipeline — orchestrates NER, classifier, extractor, summarizer.

Implements the NLPEnricher port from the domain layer.
"""

from __future__ import annotations

from rag_ddd.domain.entities import NLPEntity, NLPResult
from rag_ddd.domain.ports import NLPEnricher
from rag_ddd.infrastructure.nlp.classifier import GenAIClassifier
from rag_ddd.infrastructure.nlp.extractor import GenAIExtractor
from rag_ddd.infrastructure.nlp.ner import GenAINER
from rag_ddd.infrastructure.nlp.summarizer import GenAISummarizer


class SpacyNLPEnricher(NLPEnricher):
    """Concrete NLP enricher combining spaCy NER, keyword classification,
    regex extraction, and extractive summarization.

    Adapted from the DocuLex NLP pipeline for the Generative AI domain.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        max_summary_sentences: int = 5,
    ) -> None:
        self._ner = GenAINER(model_name=spacy_model)
        self._classifier = GenAIClassifier()
        self._extractor = GenAIExtractor()
        self._summarizer = GenAISummarizer(max_sentences=max_summary_sentences)
        # Share the NER instance with the extractor to avoid loading spaCy twice
        self._extractor._ner = self._ner

    def enrich(self, text: str) -> NLPResult:
        if not text.strip():
            return NLPResult()

        # 1. NER
        ner_result = self._ner.extract(text)
        entities = [
            NLPEntity(text=e.text, label=e.label, start=e.start, end=e.end)
            for e in ner_result.entities
        ]

        # 2. Classification
        classification = self._classifier.classify(text)

        # 3. Key info extraction
        extracted = self._extractor.extract(text)

        # 4. Summary
        summary_result = self._summarizer.summarize(text)

        return NLPResult(
            entities=entities,
            category=classification.label,
            category_confidence=classification.confidence,
            title=extracted.title,
            author=extracted.author,
            key_concepts=extracted.key_concepts,
            summary=summary_result.summary,
        )


class NoOpNLPEnricher(NLPEnricher):
    """Pass-through enricher that returns empty NLP results.

    Used when NLP enrichment is disabled.
    """

    def enrich(self, text: str) -> NLPResult:
        return NLPResult()
