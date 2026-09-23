from __future__ import annotations

from dataclasses import dataclass

MODEL_NAME = "MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli"
MAX_LENGTH = 256

# Verified against the model's real config.json (id2label) - same order
# as the current baseline model (GuardrailsAI/finetuned_nli_provenance).
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

RAW_LABEL_TO_RELATION = {
    "entailment": "SUPPORT",
    "neutral": "NOT_ENOUGH_EVIDENCE",
    "contradiction": "CONTRADICT",
}


@dataclass
class NliResult:
    raw_label: str
    predicted_relation: str
    probs: dict[str, float]
    token_count: int
    truncated: bool


class DebertaNliModel:
    """
    Wrapper around MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli
    (candidate B). Same calling convention as
    evaluation.experiments.nli_oracle.model.NliOracleModel: raw argmax
    only, premise first then hypothesis, max_length=256 to match the
    baseline for a fair comparison.
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
