"""Document classification for Generative AI domain.

Keyword-based classification into GenAI-specific categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "architecture": [
        "transformer",
        "encoder",
        "decoder",
        "attention",
        "multi-head",
        "feedforward",
        "layer norm",
        "residual",
        "positional encoding",
        "architecture",
        "neural network",
        "hidden state",
        "parameters",
    ],
    "training": [
        "fine-tuning",
        "fine-tune",
        "pre-training",
        "training",
        "loss function",
        "gradient",
        "backpropagation",
        "optimizer",
        "learning rate",
        "batch size",
        "epoch",
        "rlhf",
        "dpo",
        "lora",
        "qlora",
        "reward model",
        "alignment",
    ],
    "evaluation": [
        "benchmark",
        "evaluation",
        "metric",
        "bleu",
        "rouge",
        "perplexity",
        "accuracy",
        "human evaluation",
        "leaderboard",
        "mmlu",
        "hellaswag",
        "performance",
        "score",
    ],
    "application": [
        "chatbot",
        "rag",
        "retrieval",
        "augmented generation",
        "agent",
        "tool use",
        "function calling",
        "summarization",
        "translation",
        "code generation",
        "question answering",
        "deployment",
        "inference",
        "serving",
    ],
    "theory": [
        "embedding",
        "tokenization",
        "vocabulary",
        "softmax",
        "cross-entropy",
        "probability",
        "distribution",
        "representation",
        "latent space",
        "vector",
        "dimension",
        "cosine similarity",
        "semantic",
    ],
    "safety_ethics": [
        "hallucination",
        "bias",
        "toxicity",
        "safety",
        "guardrail",
        "alignment",
        "responsible ai",
        "fairness",
        "red teaming",
        "jailbreak",
        "prompt injection",
        "content filter",
    ],
}


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    all_scores: Dict[str, float]


class GenAIClassifier:
    """Classifies text into GenAI categories by keyword counting."""

    def __init__(self, keywords: Dict[str, List[str]] | None = None) -> None:
        self.keywords = keywords or CATEGORY_KEYWORDS

    def classify(self, text: str) -> ClassificationResult:
        text_lower = text.lower()

        scores: Dict[str, float] = {}
        for category, words in self.keywords.items():
            count = sum(1 for word in words if word in text_lower)
            scores[category] = count / len(words) if words else 0.0

        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        best_label = max(scores, key=lambda k: scores[k])
        best_score = scores[best_label]

        return ClassificationResult(
            label=best_label,
            confidence=best_score,
            all_scores=scores,
        )
