#!/usr/bin/env python3
"""Derive stepwise search metrics from an official company-news run.

No new vendor calls. Uses stored hits + locked gold cells.

  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/score_official_search_metrics.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data" / "company-news" / "official-runs" / "20260816T020806Z"
SAMPLES = ROOT / "data" / "company-news" / "samples.json"
KS = (1, 5, 10)
ENDPOINTS = (
    "exa_instant",
    "exa_deep",
    "parallel_turbo",
    "parallel_fast",
    "parallel_basic",
    "parallel_advanced",
    "firecrawl",
    "predictleads",
    "seltz_news",
    "brave",
    "tavily_advanced",
    "serp",
    "linkup_fast",
    "linkup_standard",
)


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _norm_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{parsed.path.rstrip('/')}"


def _gold_match(hit_url: str, gold_url: str) -> bool:
    if not gold_url or not hit_url:
        return False
    if _norm_url(hit_url) == _norm_url(gold_url):
        return True
    gold_host = _host(gold_url)
    gold_tail = Path(urlparse(gold_url).path).name.lower()
    hit_host = _host(hit_url)
    return bool(gold_host == hit_host and gold_tail and gold_tail in (urlparse(hit_url).path or "").lower())


def _cell_in_text(value: str, text: str) -> bool:
    raw = (value or "").lower().strip()
    if not raw:
        return True
    blob = (text or "").lower()
    token = re.sub(r"[^a-z0-9.$]+", " ", raw).strip()
    if token and token in blob:
        return True
    digits = re.sub(r"[^\d]", "", raw)
    if len(digits) >= 2 and digits in re.sub(r"[^\d]", "", blob):
        return True
    return False


def _hit_text(hit: dict[str, Any]) -> str:
    return f"{hit.get('title') or ''} {hit.get('snippet') or ''}"


def _answer_bearing(hit: dict[str, Any], cells: list[dict[str, Any]]) -> bool:
    text = _hit_text(hit)
    usable = [c for c in cells if str(c.get("value") or "").strip()]
    if not usable:
        return False
    return all(_cell_in_text(str(c.get("value") or ""), text) for c in usable)


def _bearing_ranks(block: dict[str, Any], hits: list[dict[str, Any]], cells: list[dict[str, Any]]) -> list[int]:
    judged = (block.get("ar") or {}).get("bearing_ranks")
    if isinstance(judged, list):
        ranks: list[int] = []
        for item in judged:
            try:
                rank = int(item)
            except (TypeError, ValueError):
                continue
            if rank >= 1:
                ranks.append(rank)
        return sorted(set(ranks))
    return [i for i, hit in enumerate(hits, 1) if _answer_bearing(hit, cells)]


def _tokens(text: str) -> int:
    # cl100k-style stand-in without tiktoken: ~4 chars / token on English snippets.
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return 0
    return max(1, round(len(cleaned) / 4))


def _pct(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _pctile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 1)


def score_vendor(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n = len(rows)
    correct = 0
    latencies: list[float] = []
    tokens_per_query: list[int] = []
    hits_per_query: list[int] = []
    p_at: dict[int, list[float]] = {k: [] for k in KS}
    source_p_at: dict[int, list[float]] = {k: [] for k in KS}
    ar_at = {k: 0 for k in KS}
    source_r_at = {k: 0 for k in KS}
    gold_page_and_snippet = 0
    gold_page_no_snippet = 0
    snippet_fault = 0
    extract_fault = 0
    index_miss = 0
    other_incorrect = 0
    empty_incorrect = 0

    for row in rows:
        block = (row.get("vendors") or {}).get(name) or {}
        hits = list(block.get("hits") or [])[:10]
        cells = row.get("cells") or []
        gold_url = row.get("ground_truth_url") or ""
        is_correct = bool((block.get("gold") or {}).get("correct"))
        answer = (block.get("answer") or "").strip()
        if is_correct:
            correct += 1
        latencies.append(float(block.get("latency_ms") or 0))
        tokens_per_query.append(sum(_tokens(_hit_text(h)) for h in hits))
        hits_per_query.append(len(hits))

        bearing_ranks = _bearing_ranks(block, hits, cells)
        source_ranks = [i for i, hit in enumerate(hits, 1) if _gold_match(hit["url"], gold_url)]
        gold_rank = source_ranks[0] if source_ranks else None
        gold_snippet_ok = bool(gold_rank and gold_rank in bearing_ranks)
        any_bearing = bool(bearing_ranks)
        page_in_10 = gold_rank is not None

        if page_in_10 and gold_snippet_ok:
            gold_page_and_snippet += 1
        if page_in_10 and not gold_snippet_ok:
            gold_page_no_snippet += 1

        if not is_correct:
            if page_in_10 and not gold_snippet_ok:
                snippet_fault += 1
            elif any_bearing:
                extract_fault += 1
            elif not page_in_10:
                index_miss += 1
            else:
                other_incorrect += 1
            if not answer:
                empty_incorrect += 1

        for k in KS:
            top = hits[:k]
            p_at[k].append(sum(1 for rank in bearing_ranks if rank <= k) / k)
            source_p_at[k].append(sum(1 for hit in top if _gold_match(hit["url"], gold_url)) / k)
            if any(rank <= k for rank in bearing_ranks):
                ar_at[k] += 1
            if gold_rank is not None and gold_rank <= k:
                source_r_at[k] += 1

    incorrect = n - correct
    return {
        "n": n,
        "accuracy": _pct(correct, n),
        "correct": correct,
        "incorrect": incorrect,
        "latency_ms_mean": _mean(latencies),
        "latency_ms_p50": _pctile(latencies, 0.5),
        "latency_ms_p95": _pctile(latencies, 0.95),
        "hits_mean": _mean([float(x) for x in hits_per_query]),
        "snippet_tokens_mean": _mean([float(x) for x in tokens_per_query]),
        "snippet_tokens_p50": _pctile([float(x) for x in tokens_per_query], 0.5),
        "snippet_tokens_per_hit": round(
            (sum(tokens_per_query) / sum(hits_per_query)) if sum(hits_per_query) else 0.0, 1
        ),
        "precision_at": {str(k): _mean(p_at[k]) for k in KS},
        "source_precision_at": {str(k): _mean(source_p_at[k]) for k in KS},
        "answer_bearing_recall_at": {str(k): _pct(ar_at[k], n) for k in KS},
        "source_recall_at": {str(k): _pct(source_r_at[k], n) for k in KS},
        "gold_page_snippet_ok": _pct(gold_page_and_snippet, n),
        "gold_page_snippet_miss": _pct(gold_page_no_snippet, n),
        "snippet_fault": {
            "n": snippet_fault,
            "rate": _pct(snippet_fault, n),
            "share_of_errors": _pct(snippet_fault, incorrect),
        },
        "extract_fault": {
            "n": extract_fault,
            "rate": _pct(extract_fault, n),
            "share_of_errors": _pct(extract_fault, incorrect),
        },
        "index_miss": {
            "n": index_miss,
            "rate": _pct(index_miss, n),
            "share_of_errors": _pct(index_miss, incorrect),
        },
        "empty_incorrect": {
            "n": empty_incorrect,
            "rate": _pct(empty_incorrect, n),
            "share_of_errors": _pct(empty_incorrect, incorrect),
        },
        "other_incorrect": {
            "n": other_incorrect,
            "rate": _pct(other_incorrect, n),
        },
    }


def main() -> None:
    payload = json.loads((RUN / "run.json").read_text())
    samples = {s["id"]: s for s in json.loads(SAMPLES.read_text())}
    rows = []
    for case in payload["cases"]:
        sample = samples.get(case["id"])
        if not sample:
            continue
        row = dict(case)
        row["cells"] = sample.get("cells") or (sample.get("gold") or {}).get("cells") or []
        row["ground_truth_url"] = case.get("ground_truth_url") or (sample.get("gold") or {}).get("primary_url") or ""
        rows.append(row)

    vendors = {name: score_vendor(rows, name) for name in ENDPOINTS if any(name in (r.get("vendors") or {}) for r in rows)}
    out = {
        "created_from": str(RUN / "run.json"),
        "n": len(rows),
        "definitions": {
            "accuracy": "Judge: extracted answer contains every gold cell.",
            "precision_at_k": "Mean over queries of (# top-K hits whose snippet contains every gold cell) / K.",
            "answer_bearing_recall_at_k": "Share of queries with at least one top-K hit the LLM judge scored as already containing every gold cell.",
            "source_recall_at_k": "Share of queries where the locked gold URL appears in top K.",
            "snippet_fault": "Incorrect, gold page retrieved, and the LLM judge did not score that page's snippet as containing the gold cell.",
            "extract_fault": "Incorrect, but the LLM judge scored some returned snippet as already containing the gold cell.",
            "index_miss": "Incorrect, gold page not retrieved, and no answer-bearing snippet.",
            "snippet_tokens": "Sum of title+snippet tokens per query. Token ≈ 4 characters (no tiktoken).",
        },
        "vendors": vendors,
    }
    dest = RUN / "metrics.search.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")

    def pct(v: float) -> str:
        return f"{v:.1%}"

    print(f"n={len(rows)} wrote {dest}")
    print()
    print(f"{'endpoint':20} acc   AR@1  AR@5  AR@10  P@5   srcR@5  snip_fault  extract  idx_miss  lat_ms  tok")
    for name, block in vendors.items():
        print(
            f"{name:20} {pct(block['accuracy']):5} "
            f"{pct(block['answer_bearing_recall_at']['1']):5} "
            f"{pct(block['answer_bearing_recall_at']['5']):5} "
            f"{pct(block['answer_bearing_recall_at']['10']):6} "
            f"{pct(block['precision_at']['5']):5} "
            f"{pct(block['source_recall_at']['5']):6} "
            f"{pct(block['snippet_fault']['rate']):10} "
            f"{pct(block['extract_fault']['rate']):7} "
            f"{pct(block['index_miss']['rate']):8} "
            f"{block['latency_ms_mean']:7.0f} "
            f"{block['snippet_tokens_mean']:5.0f}"
        )


if __name__ == "__main__":
    main()
