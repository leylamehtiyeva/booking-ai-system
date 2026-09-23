from __future__ import annotations

from dataclasses import dataclass

MODEL_NAME = "GuardrailsAI/finetuned_nli_provenance"
MAX_LENGTH = 256

# Verified against the model's real config.json (id2label), not assumed.
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

RAW_LABEL_TO_RELATION = {
    "entailment": "SUPPORT",
    "neutral": "RELEVANT_NEUTRAL",
    "contradiction": "CONTRADICT",
}


@dataclass
class NliResult:
    raw_label: str
    predicted_relation: str
    probs: dict[str, float]
    token_count: int
    truncated: bool


class NliOracleModel:
    """
    Thin wrapper around GuardrailsAI/finetuned_nli_provenance.

    Runs raw argmax classification only - no confidence threshold is
    applied here. premise/hypothesis order and max_length follow the
    convention documented by the base model
    (ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli):
    tokenizer(premise, hypothesis, truncation=True, max_length=256).
    """

    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self._torch = torch

        config_id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if config_id2label != ID2LABEL:
            raise RuntimeError(
                f"Model id2label changed since design-time verification: "
                f"expected {ID2LABEL}, got {config_id2label}"
            )

    def predict(self, premise: str, hypothesis: str) -> NliResult:
        torch = self._torch

        # Token count is measured without truncation so we can report
        # the real pre-truncation length, then truncated=True/False
        # reflects whether max_length actually clipped this pair.
        untruncated = self.tokenizer(premise, hypothesis, truncation=False)
        token_count = len(untruncated["input_ids"])
        truncated = token_count > self.max_length

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        pred_id = int(torch.argmax(logits).item())
        raw_label = ID2LABEL[pred_id]

        return NliResult(
            raw_label=raw_label,
            predicted_relation=RAW_LABEL_TO_RELATION[raw_label],
            probs={ID2LABEL[i]: round(probs[i], 6) for i in range(3)},
            token_count=token_count,
            truncated=truncated,
        )
