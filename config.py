"""
Shared configuration for the RDD ingestion -> filtering -> classification pipeline.

Maps directly onto pipeline_requirements.md (P1, P2, P3 / CORE-3, CORE-4).
"""

# --- P1: Ingestion ---------------------------------------------------------
SIGNAL_APP_ID = "org.thoughtcrime.securesms"   # Signal on Google Play
REQUIRED_FIELDS = ["text", "rating", "date", "version"]  # P1.1 minimum fields

# --- P2: Filtering -----------------------------------------------------------
MIN_WORD_COUNT = 4                 # P2.2 text quality threshold
SPAM_REPEAT_CHAR_RATIO = 0.4       # P2.3 fraction of a review that is one repeated char
SPAM_PROMO_KEYWORDS = [
    "click here", "visit our website", "free gift", "subscribe now",
    "promo code", "www.", "http://", "https://", "discount code",
]
ALLOWED_LANGUAGE = "en"            # P2.1 — non-English reviews are deferred

# --- P3: Classification ------------------------------------------------------
LABELS = ["bug-report", "feature-request", "ux-complaint", "non-actionable"]
UNCLASSIFIED_LABEL = "unclassified"
CONFIDENCE_THRESHOLD = 0.55        # below this -> UNCLASSIFIED_LABEL (P3.2)

# Zero-shot hypothesis template for the NLI-based placeholder classifier.
# roberta-large-mnli scores each label as an entailment hypothesis against
# the review text (the premise).
ZERO_SHOT_HYPOTHESIS_TEMPLATE = "This app review is an example of a {}."
LABEL_DESCRIPTIONS = {
    "bug-report": "bug report describing something broken or crashing",
    "feature-request": "feature request asking for new or missing functionality",
    "ux-complaint": "complaint about confusing or frustrating user experience or design",
    "non-actionable": "generic praise or complaint with no specific actionable concern",
}

# Placeholder classifier model now; swap for your fine-tuned checkpoint later.
ZERO_SHOT_MODEL_NAME = "roberta-large-mnli"
FINE_TUNED_MODEL_PATH = None  # set to a local path once Phase 2 fine-tuning is done

# --- Logging -----------------------------------------------------------------
DEFERRED_LOG_PATH = "logs/deferred_reviews.jsonl"
