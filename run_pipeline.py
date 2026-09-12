"""
Runs P1 (ingestion) -> P2 (filtering) -> P3 (classification) end-to-end.

By default this loads the bundled synthetic sample (data/sample_signal_reviews.json)
so it runs anywhere, with no network required. To run against real, live
Signal reviews instead (on a machine with network access to Google Play):

    python run_pipeline.py --live --count 200

Output: prints a summary, and writes results/classified_reviews.json +
logs/deferred_reviews.jsonl (P2.4 / traceability of what got set aside).
"""
import argparse
import json
import os
from collections import Counter

from p1_ingestion import ingest, fetch_reviews
from p2_filtering import filter_reviews
from p3_classification import classify_reviews, get_classifier

SAMPLE_PATH = "data/sample_signal_reviews.json"
OUTPUT_PATH = "results/classified_reviews.json"


def load_sample(path: str = SAMPLE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(use_live: bool, count: int):
    if use_live:
        print("Fetching live reviews from Google Play (requires network)...")
        raw_reviews = fetch_reviews(count=count)
    else:
        print(f"Loading synthetic sample from {SAMPLE_PATH} (no network needed)...")
        raw_reviews = load_sample()

    print(f"\n--- P1: Ingestion ---")
    ingested = ingest(raw_reviews)
    print(f"Input: {len(raw_reviews)} raw reviews -> {len(ingested)} after version "
          f"normalisation + dedup ({len(raw_reviews) - len(ingested)} exact duplicates removed)")

    print(f"\n--- P2: Filtering ---")
    working_set, filter_deferred = filter_reviews(ingested)
    reasons = Counter(d["deferred_reason"] for d in filter_deferred)
    print(f"Working set: {len(working_set)} | Deferred: {len(filter_deferred)} "
          f"({dict(reasons)})")

    print(f"\n--- P3: Classification ---")
    classifier = get_classifier()
    print(f"Using classifier: {classifier.__class__.__name__}")
    actionable, class_deferred = classify_reviews(working_set, classifier)
    label_counts = Counter(r["label"] for r in actionable)
    deferred_reasons = Counter(r["deferred_reason"] for r in class_deferred)
    print(f"Actionable tickets input: {len(actionable)} {dict(label_counts)}")
    print(f"Deferred (non-actionable/unclassified): {len(class_deferred)} {dict(deferred_reasons)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(actionable, f, indent=2)

    print(f"\nWrote {len(actionable)} actionable, classified reviews -> {OUTPUT_PATH}")
    print(f"Full deferred-set log (P2.4 + P3.3, all reasons combined) -> logs/deferred_reviews.jsonl")

    print("\n--- Sample actionable reviews ---")
    for r in actionable[:5]:
        print(f"  [{r['label']:16s} conf={r['confidence']:.2f}] {r['text'][:70]}")

    return actionable, filter_deferred + class_deferred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                         help="Fetch real reviews from Google Play instead of the sample file")
    parser.add_argument("--count", type=int, default=200,
                         help="Number of live reviews to fetch (only with --live)")
    args = parser.parse_args()
    run(use_live=args.live, count=args.count)
