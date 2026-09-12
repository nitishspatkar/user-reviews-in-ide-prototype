"""
P2 — Filtering (pipeline_requirements.md)

P2.1  Language detection   -> is_english()
P2.2  Text quality check   -> passes_text_quality()
P2.3  Spam detection       -> is_spam()
P2.4  Deferred set logging -> log_deferred()

filter_reviews() ties these together and returns (working_set, deferred_set),
per CORE-3: both groups are produced and both are logged.
"""
import json
import os
import re
from collections import Counter
from typing import List, Dict, Any, Tuple

from config import (
    MIN_WORD_COUNT, SPAM_REPEAT_CHAR_RATIO, SPAM_PROMO_KEYWORDS,
    ALLOWED_LANGUAGE, DEFERRED_LOG_PATH, REQUIRED_FIELDS,
)
from p1_ingestion import check_required_fields

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0  # deterministic results (STUDY-5 reproducibility)
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False


def is_english(text: str) -> bool:
    """P2.1 — Detect language; only English reviews enter the working set."""
    if not text or not text.strip():
        return False
    if not _HAS_LANGDETECT:
        # Fallback heuristic if langdetect isn't installed: ASCII-letter ratio.
        letters = re.findall(r"[A-Za-z]", text)
        return len(letters) / max(len(text), 1) > 0.6
    try:
        return detect(text) == ALLOWED_LANGUAGE
    except LangDetectException:
        return False


def passes_text_quality(text: str) -> bool:
    """P2.2 — Reject star-only submissions and reviews under the minimum
    word-count threshold.
    """
    if not text or not text.strip():
        return False
    word_count = len(text.strip().split())
    return word_count >= MIN_WORD_COUNT


def is_spam(text: str) -> bool:
    """P2.3 — Heuristic spam flag: dominant repeated character, or a
    promotional keyword/URL.
    """
    if not text:
        return False
    lowered = text.lower()
    if any(kw in lowered for kw in SPAM_PROMO_KEYWORDS):
        return True
    counts = Counter(text.replace(" ", ""))
    if counts:
        most_common_char, freq = counts.most_common(1)[0]
        if freq / max(len(text.replace(" ", "")), 1) >= SPAM_REPEAT_CHAR_RATIO:
            return True
    return False


def _defer(review: Dict[str, Any], reason: str) -> Dict[str, Any]:
    deferred = dict(review)
    deferred["deferred_reason"] = reason
    return deferred


def log_deferred(deferred_set: List[Dict[str, Any]], path: str = DEFERRED_LOG_PATH) -> None:
    """P2.4 — Append the deferred set to a JSONL log (never discarded)."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        for d in deferred_set:
            f.write(json.dumps({
                "review_id": d.get("review_id"),
                "text": d.get("text"),
                "date": d.get("date"),
                "reason": d.get("deferred_reason"),
            }) + "\n")


def filter_reviews(reviews: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run P2.1 -> P2.4. Returns (working_set, deferred_set).

    Order of checks: missing-fields -> low-text -> suspected-spam -> non-english.
    Length/spam heuristics run before language detection on purpose:
    langdetect is unreliable on very short strings (e.g. "ok", "meh", or a
    repeated-character spam string can get misdetected as non-English), so
    catching those with the cheaper checks first gives more accurate reason
    tags in the deferred-set log.
    """
    working_set, deferred_set = [], []
    for r in reviews:
        text = r.get("text", "") or ""

        if not check_required_fields(r):
            deferred_set.append(_defer(r, "missing-fields"))
            continue
        if not passes_text_quality(text):
            deferred_set.append(_defer(r, "low-text"))
            continue
        if is_spam(text):
            deferred_set.append(_defer(r, "suspected-spam"))
            continue
        if not is_english(text):
            deferred_set.append(_defer(r, "non-english"))
            continue

        working_set.append(r)

    log_deferred(deferred_set)
    return working_set, deferred_set


if __name__ == "__main__":
    sample = [
        {"review_id": "1", "text": "The app crashes every time I open a group chat", "rating": 1, "date": "2024-01-01", "version": "6.1.0"},
        {"review_id": "2", "text": "Bien", "rating": 5, "date": "2024-01-02", "version": "6.1.0"},
        {"review_id": "3", "text": "great", "rating": 5, "date": "2024-01-02", "version": "6.1.0"},
        {"review_id": "4", "text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "rating": 3, "date": "2024-01-03", "version": "6.1.0"},
    ]
    w, d = filter_reviews(sample)
    print("working:", w)
    print("deferred:", d)
