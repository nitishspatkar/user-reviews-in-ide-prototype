"""
P1 — Ingestion (pipeline_requirements.md)

P1.1  API retrieval      -> fetch_reviews()
P1.2  Version normalisation -> normalize_version()
P1.3  Deduplication      -> deduplicate()

fetch_reviews() uses google-play-scraper and needs real network access to
play.google.com. It will not run inside this sandbox (network is restricted
to dev package registries) but is written to run as-is on your machine:

    pip install google-play-scraper
    python -c "from p1_ingestion import fetch_reviews; print(fetch_reviews(count=20))"
"""
import re
import json
import time
import socket
from typing import List, Dict, Any
socket.setdefaulttimeout(30)
from config import SIGNAL_APP_ID, REQUIRED_FIELDS


def fetch_reviews(app_id: str = SIGNAL_APP_ID, count: int = 200,
                   lang: str = "en", country: str = "us") -> List[Dict[str, Any]]:
    """P1.1 — Fetch raw reviews from the Google Play Store.

    Returns a list of dicts with (at least) the fields the API provides.
    Requires network access to play.google.com — run this locally, not
    inside a network-restricted sandbox.
    """
    try:
        from google_play_scraper import Sort, reviews
    except ImportError as e:
        raise ImportError(
            "google-play-scraper is not installed. Run:\n"
            "    pip install google-play-scraper"
        ) from e

    #raw = reviews_all(app_id, lang=lang, country=country, sort=Sort.NEWEST)

    raw = []
    token = None
    batch_size = 100  # reviews per page
    while len(raw) < count:
        remaining = count - len(raw)
        result, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=min(batch_size, remaining),
            continuation_token=token,
        )
        if not result:
            break
        raw.extend(result)
        if token is None:
            break
        time.sleep(1)  # brief pause between pages to avoid throttling


    out = []
    for r in raw[:count] if count else raw:
        out.append({
            "review_id": r.get("reviewId"),
            "text": r.get("content"),
            "rating": r.get("score"),
            "date": r.get("at").isoformat() if r.get("at") else None,
            "version": r.get("reviewCreatedVersion") or r.get("appVersion"),
            "thumbs_up": r.get("thumbsUpCount"),
        })
    return out


def check_required_fields(review: Dict[str, Any]) -> bool:
    """P1.1 — Reviews missing any of the minimum required fields are deferred
    (with reason tag 'missing-fields') at the filtering stage, not dropped here.
    """
    return all(review.get(f) not in (None, "") for f in REQUIRED_FIELDS)


_VERSION_RE = re.compile(r"(\d+)(?:[._](\d+))?(?:[._](\d+))?")


def normalize_version(raw_version: str) -> str:
    """P1.2 — Normalise a free-form version string into 'major.minor.patch'.

    Examples: 'v2.3.1-beta' -> '2.3.1', '2.3' -> '2.3.0', '2' -> '2.0.0'.
    Returns '' if no version-like pattern is found (caller should treat the
    review as missing-fields).
    """
    if not raw_version:
        return ""
    match = _VERSION_RE.search(str(raw_version))
    if not match:
        return ""
    major, minor, patch = match.groups()
    major = major or "0"
    minor = minor or "0"
    patch = patch or "0"
    return f"{major}.{minor}.{patch}"


def deduplicate(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """P1.3 — Remove exact duplicates (same text, date, version), which can
    appear due to API pagination or repeated fetches.
    """
    seen = set()
    deduped = []
    for r in reviews:
        key = (r.get("text"), r.get("date"), r.get("version"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def ingest(raw_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run P1.2 -> P1.3 over an already-fetched batch of raw reviews.
    (P1.1 / fetch_reviews is kept separate since it needs live network access.)
    """
    processed = []
    for r in raw_reviews:
        r = dict(r)
        r["version"] = normalize_version(r.get("version", ""))
        processed.append(r)
    return deduplicate(processed)


if __name__ == "__main__":
    # Local smoke test using fetch_reviews() — requires network access.
    print("fetching reviews")
    reviews = fetch_reviews(count=50)
    reviews = ingest(reviews)
    print(json.dumps(reviews[:5], indent=2))
    print(f"Fetched and normalised {len(reviews)} reviews.")
