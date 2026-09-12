"""
P3 — Classification (pipeline_requirements.md)

P3.1  Label assignment     -> RoBERTaClassifier.classify()
P3.2  Confidence scoring   -> included in classify() output; low confidence -> UNCLASSIFIED_LABEL
P3.3  Non-actionable exit  -> classify_reviews() splits working set from deferred set

Classifier backends (same interface, so you can swap one line later):

  ZeroShotRoBERTaClassifier   <- POC placeholder, used now.
      Uses roberta-large-mnli as an NLI model in zero-shot mode: each label
      description is scored as an entailment hypothesis against the review
      text, no training data required. This matches the Phase-3 POC plan
      (placeholder classifier before the fine-tuned model exists).

  FineTunedRoBERTaClassifier  <- swap in once Phase 2 fine-tuning is done.
      Loads a locally fine-tuned roberta-base checkpoint
      (trained on Maalej & Nabil, per the thesis proposal) and does direct
      5-way classification instead of zero-shot entailment.

  MockClassifier              <- offline fallback, demo/testing only.
      Keyword-heuristic stand-in used automatically when transformers/torch
      or a network path to Hugging Face is unavailable (e.g. this sandbox),
      so the pipeline can still be run and inspected end-to-end. Never use
      this for real evaluation numbers.

Run locally (with real network + transformers installed):
    pip install transformers torch
    python p3_classification.py
"""
from typing import List, Dict, Any, Tuple
import re

from config import (
    LABELS, UNCLASSIFIED_LABEL, CONFIDENCE_THRESHOLD,
    ZERO_SHOT_MODEL_NAME, ZERO_SHOT_HYPOTHESIS_TEMPLATE, LABEL_DESCRIPTIONS,
    FINE_TUNED_MODEL_PATH,
)


class BaseClassifier:
    def classify(self, text: str) -> Tuple[str, float]:
        """Return (label, confidence) where label is one of LABELS or
        UNCLASSIFIED_LABEL."""
        raise NotImplementedError


class ZeroShotRoBERTaClassifier(BaseClassifier):
    """Placeholder classifier: roberta-large-mnli used zero-shot via NLI
    entailment scoring. Requires `transformers` + `torch` and network access
    to download the model weights the first time.
    """

    def __init__(self, model_name: str = ZERO_SHOT_MODEL_NAME):
        from transformers import pipeline  # deferred import: heavy dependency
        self._pipe = pipeline("zero-shot-classification", model=model_name)
        self._candidate_labels = [LABEL_DESCRIPTIONS[l] for l in LABELS]
        self._label_lookup = {v: k for k, v in LABEL_DESCRIPTIONS.items()}

    def classify(self, text: str) -> Tuple[str, float]:
        result = self._pipe(
            text,
            candidate_labels=self._candidate_labels,
            hypothesis_template=ZERO_SHOT_HYPOTHESIS_TEMPLATE,
        )
        top_desc, top_score = result["labels"][0], result["scores"][0]
        label = self._label_lookup[top_desc]
        if top_score < CONFIDENCE_THRESHOLD:
            return UNCLASSIFIED_LABEL, top_score
        return label, top_score


class FineTunedRoBERTaClassifier(BaseClassifier):
    """Direct 5-way classification with a locally fine-tuned roberta-base
    checkpoint. Point FINE_TUNED_MODEL_PATH (config.py) at your checkpoint
    directory once Phase 2 fine-tuning is complete.
    """

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        # Expect the checkpoint's id2label to match LABELS ordering; adjust
        # this mapping at training time if your label set differs.
        self._id2label = self._model.config.id2label

    def classify(self, text: str) -> Tuple[str, float]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        top_idx = int(self._torch.argmax(probs))
        confidence = float(probs[top_idx])
        label = self._id2label[top_idx]
        if confidence < CONFIDENCE_THRESHOLD:
            return UNCLASSIFIED_LABEL, confidence
        return label, confidence


class MockClassifier(BaseClassifier):
    """Offline, keyword-heuristic stand-in used only when no real model can
    be loaded (no network / no transformers installed). Lets the pipeline
    run end-to-end for structural testing. NOT a research-grade classifier.
    """

    _RULES = [
        ("bug-report", ["crash", "crashes", "bug", "freeze", "frozen", "error", "broken", "doesn't work", "not working", "fails"]),
        ("feature-request", ["wish", "please add", "would be nice", "feature request", "add support", "missing", "should have", "could you add"]),
        ("ux-complaint", ["confusing", "hard to", "difficult to", "annoying", "cluttered", "hate the", "ugly", "clunky", "unintuitive"]),
    ]

    def classify(self, text: str) -> Tuple[str, float]:
        lowered = text.lower()
        for label, keywords in self._RULES:
            if any(kw in lowered for kw in keywords):
                return label, 0.75
        # Praise / generic sentiment with no specific concern
        if re.search(r"\b(love|great|awesome|amazing|excellent|good app)\b", lowered):
            return "non-actionable", 0.8
        return UNCLASSIFIED_LABEL, 0.3


def get_classifier() -> BaseClassifier:
    """Selects a backend: fine-tuned checkpoint if configured, else the
    zero-shot RoBERTa placeholder, else the offline mock (with a warning).
    """
    if FINE_TUNED_MODEL_PATH:
        try:
            return FineTunedRoBERTaClassifier(FINE_TUNED_MODEL_PATH)
        except Exception as e:
            print(f"[warn] could not load fine-tuned checkpoint ({e}); falling back.")
    try:
        return ZeroShotRoBERTaClassifier()
    except Exception as e:
        print(f"[warn] could not load {ZERO_SHOT_MODEL_NAME} "
              f"(likely no network/transformers here) — using MockClassifier "
              f"for this demo run only. Real runs should use the RoBERTa "
              f"backend. Reason: {e}")
        return MockClassifier()


def classify_reviews(reviews: List[Dict[str, Any]],
                      classifier: BaseClassifier = None
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """P3.1 -> P3.3. Returns (working_set, deferred_set).

    working_set:  reviews labeled bug-report / feature-request / ux-complaint,
                  each with 'label' and 'confidence' attached.
    deferred_set: reviews labeled non-actionable or unclassified, tagged with
                  a deferred_reason for the deferred-set log.
    """
    classifier = classifier or get_classifier()
    working_set, deferred_set = [], []

    for r in reviews:
        label, confidence = classifier.classify(r.get("text", ""))
        r = dict(r)
        r["label"] = label
        r["confidence"] = round(confidence, 3)

        if label in ("non-actionable", UNCLASSIFIED_LABEL):
            r["deferred_reason"] = label
            deferred_set.append(r)
        else:
            working_set.append(r)

    return working_set, deferred_set


if __name__ == "__main__":
    sample_texts = [
        "The app crashes every time I open a group chat on Android 14",
        "Please add a way to schedule messages, would be so useful",
        "The settings menu is so confusing, I can never find backup options",
        "Love this app, best messenger out there",
        "meh",
    ]
    clf = get_classifier()
    for t in sample_texts:
        label, conf = clf.classify(t)
        print(f"{conf:.2f}  {label:16s}  {t}")
