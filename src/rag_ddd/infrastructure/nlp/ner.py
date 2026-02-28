"""Named Entity Recognition for Generative AI domain.

Combines two approaches:
1. spaCy (ML): generic entities (PERSON, ORG, DATE)
2. Regex: domain-specific entities (MODEL, TECHNIQUE, PAPER_REF, METRIC)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# --- Regex patterns for GenAI domain entities ---

MODEL_PATTERNS = [
    r"\b(?:GPT-[234]\w*|ChatGPT|DALL[-·]?E(?:\s*[23])?|Codex)\b",
    r"\bClaude(?:\s+\d[\w.]*)?(?:\s+(?:Opus|Sonnet|Haiku))?\b",
    r"\b(?:Gemini|Gemma|PaLM(?:\s*2)?|Bard|Imagen)\b",
    r"\b(?:LLaMA|Llama)\s*[23]?(?:\.\d+)?\b",
    r"\b(?:Mistral|Mixtral)(?:\s+\w+)?\b",
    r"\b(?:BERT|RoBERTa|DeBERTa|XLNet|ELECTRA|ALBERT)\b",
    r"\b(?:T5|FLAN-T5|UL2|PaLM-E)\b",
    r"\b(?:Stable\s*Diffusion|Midjourney|FLUX)\b",
    r"\b(?:Whisper|CLIP|BLIP(?:-2)?|SAM|Segment\s+Anything)\b",
    r"\b(?:Phi-[123]|Qwen\d*|DeepSeek(?:-\w+)?|Yi-\w+)\b",
    r"\b(?:Falcon|MPT|StableLM|Vicuna|Alpaca)\b",
]

TECHNIQUE_PATTERNS = [
    r"\b(?:LoRA|QLoRA|DoRA|AdaLoRA)\b",
    r"\b(?:RLHF|DPO|PPO|RLAIF|Constitutional\s+AI)\b",
    r"\b(?:RAG|Retrieval[\s-]Augmented\s+Generation)\b",
    r"\b(?:Chain[\s-]of[\s-]Thought|CoT|Tree[\s-]of[\s-]Thought|ToT)\b",
    r"\b(?:In[\s-]?Context\s+Learning|ICL|Few[\s-]?Shot|Zero[\s-]?Shot)\b",
    r"\bfine[\s-]?tun(?:e|ing|ed)\b",
    r"\b(?:knowledge\s+distillation|model\s+pruning|quantization)\b",
    r"\b(?:attention\s+mechanism|self[\s-]attention|cross[\s-]attention|multi[\s-]head\s+attention)\b",
    r"\b(?:transformer|encoder[\s-]decoder)\b",
    r"\b(?:tokenization|BPE|WordPiece|SentencePiece)\b",
    r"\b(?:positional\s+encoding|rotary\s+embedding|RoPE|ALiBi)\b",
    r"\b(?:flash\s+attention|grouped[\s-]query\s+attention|GQA|MQA)\b",
    r"\b(?:mixture[\s-]of[\s-]experts|MoE|sparse\s+MoE)\b",
    r"\b(?:embedding|word2vec|GloVe|fastText)\b",
]

PAPER_REF_PATTERNS = [
    r"(?:Vaswani|Devlin|Brown|Radford|Touvron|Raffel|Lewis)\s+et\s+al\.?\s*(?:\(?\d{4}\)?)?",
    r'"[A-Z][^"]{10,80}"\s*\(\d{4}\)',
    r"\b(?:arXiv|arxiv):\d{4}\.\d{4,5}(?:v\d+)?\b",
]

METRIC_PATTERNS = [
    r"\b(?:BLEU|ROUGE(?:-[LN12])?|METEOR|BERTScore|BLEURT)\b",
    r"\b(?:perplexity|PPL|cross[\s-]entropy\s+loss)\b",
    r"\b(?:F1[\s-]score|accuracy|precision|recall|AUC[\s-]ROC)\b",
    r"\b(?:MMLU|HellaSwag|ARC|TruthfulQA|GSM8K|HumanEval)\b",
]


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass
class NERResult:
    entities: List[Entity] = field(default_factory=list)

    def get_by_label(self, label: str) -> List[Entity]:
        return [e for e in self.entities if e.label == label]

    @property
    def persons(self) -> List[Entity]:
        return self.get_by_label("PERSON")

    @property
    def organizations(self) -> List[Entity]:
        return self.get_by_label("ORG")

    @property
    def models(self) -> List[Entity]:
        return self.get_by_label("MODEL")

    @property
    def techniques(self) -> List[Entity]:
        return self.get_by_label("TECHNIQUE")


class GenAINER:
    """Entity extractor: spaCy (ML) + regex rules (GenAI domain)."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self._model_name = model_name
        self._nlp = None

    def _get_nlp(self):
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self._model_name)
        return self._nlp

    def extract(self, text: str) -> NERResult:
        result = NERResult()

        # 1. spaCy ML entities (PERSON, ORG, DATE, etc.)
        nlp = self._get_nlp()
        doc = nlp(text)
        for ent in doc.ents:
            result.entities.append(
                Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            )

        # 2. Domain-specific regex entities
        result.entities.extend(self._extract_by_patterns(text, MODEL_PATTERNS, "MODEL"))
        result.entities.extend(self._extract_by_patterns(text, TECHNIQUE_PATTERNS, "TECHNIQUE"))
        result.entities.extend(self._extract_by_patterns(text, PAPER_REF_PATTERNS, "PAPER_REF"))
        result.entities.extend(self._extract_by_patterns(text, METRIC_PATTERNS, "METRIC"))

        return result

    @staticmethod
    def _extract_by_patterns(text: str, patterns: List[str], label: str) -> List[Entity]:
        entities: List[Entity] = []
        seen_spans: set[tuple[int, int]] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append(
                        Entity(text=match.group(), label=label, start=span[0], end=span[1])
                    )
        return entities
