# RDD Pipeline — P1 Ingestion, P2 Filtering, P3 Classification

Implements the first three stages of `pipeline_requirements.md`: fetch Signal
reviews, clean/filter them, classify for actionability. P4 (repo linkage),
P5 (prioritisation) and P6 (ticket view/lifecycle) are not built yet.

## Files

| File | Maps to | Purpose |
|---|---|---|
| `config.py` | — | labels, thresholds, model names, shared constants |
| `p1_ingestion.py` | P1.1–P1.3 | fetch from Google Play, normalise version, dedup |
| `p2_filtering.py` | P2.1–P2.4 | language, text-quality, spam filters + deferred log |
| `p3_classification.py` | P3.1–P3.3 | RoBERTa classifier (zero-shot now, fine-tuned slot later) |
| `run_pipeline.py` | — | orchestrates P1 → P2 → P3, prints a summary |
| `data/sample_signal_reviews.json` | — | synthetic sample covering every branch (bug/feature/ux/praise, non-English, spam, short text, dupes, missing fields) |

## Quick start (offline demo, what was run to produce these results)

```bash
python run_pipeline.py
```

Loads the bundled sample, runs it through all three stages, writes
`results/classified_reviews.json` and `logs/deferred_reviews.jsonl`.
No network needed — this is what was tested in this session.

## Running for real, on your machine

```bash
pip install -r requirements.txt
python run_pipeline.py --live --count 200
```

`--live` calls `fetch_reviews()` in `p1_ingestion.py`, which hits the Google
Play Store for `org.thoughtcrime.securesms` (Signal) via `google-play-scraper`.
This needs real network access — it will not run inside a network-restricted
sandbox, which is why the demo here used the sample file instead.

## Classifier status — important

`get_classifier()` in `p3_classification.py` tries three backends in order:

1. **`FineTunedRoBERTaClassifier`** — used only if `FINE_TUNED_MODEL_PATH` in
   `config.py` points at a real checkpoint. Not set yet — this is the
   roberta-base checkpoint fine-tuned on Maalej & Nabil from Phase 2 of the
   thesis plan. Point this at it once that training is done and everything
   downstream switches over with no other code changes.
2. **`ZeroShotRoBERTaClassifier`** — the current default. Uses
   `roberta-large-mnli` zero-shot (entailment scoring against each label's
   description), matching the Phase-3 POC plan of using a placeholder
   classifier before the fine-tuned model exists. Needs `transformers` +
   `torch` and network access to Hugging Face to download weights.
3. **`MockClassifier`** — automatic fallback if neither of the above can
   load (e.g. this sandbox, which has no route to huggingface.co). It's a
   keyword-heuristic stand-in that keeps the pipeline runnable end-to-end
   for structural testing. **The results in `results/classified_reviews.json`
   from this session used the mock classifier — treat the labels there as a
   pipeline-wiring check, not real classification output.**

To actually exercise the RoBERTa zero-shot path, run `run_pipeline.py`
somewhere with `transformers`/`torch` installed and Hugging Face Hub
reachable.

## What's deliberately not done yet

- **P4 (repo linkage, staleness)**, **P5 (prioritisation)**, **P6 (ticket
  view/lifecycle)** — next stages, not part of this request.
- **Fine-tuned RoBERTa checkpoint** — separate Phase-2 training effort per
  the thesis timeline; this pipeline is ready to consume it once it exists.
- **VS Code / Cursor integration** — deferred per your last answer; this is
  scripts + JSON output only for now.
