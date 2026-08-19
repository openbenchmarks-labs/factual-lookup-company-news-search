#!/usr/bin/env python3
"""Single-query atomic company-news probe.

One natural-language query, one API request, max 10 results, no rewrite,
no page fetch. Vendors: Exa, Parallel (turbo), Firecrawl, PredictLeads
(company-domain news events, category-filtered).

  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/run_atomic_news_probe.py
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/run_atomic_news_probe.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _shared import load_environment  # noqa: E402
from websearch.predictleads_news import (  # noqa: E402
    expand_categories,
    fetch_category_events,
    pick_categories,
    recipe_categories,
)

SAMPLES = ROOT / "data" / "company-news" / "samples.json"
VENDORS = ("exa", "parallel", "firecrawl", "predictleads")
DEFAULT_MODEL = "gpt-5.6-terra"
MAX_RESULTS = 10
SNIPPET_CHARS = 1200

# Published list prices as of 2026-08-15. Firecrawl is 2 credits / 10 results;
# USD uses a mid-plan $0.0025/credit so one 10-result search is $0.005.
UNIT_COST_USD = {
    "exa": 0.007,
    "parallel": 0.001,
    "firecrawl": 0.005,
    "predictleads": 0.04,
    "openai_input_per_m": 0.40,
    "openai_output_per_m": 1.60,
}


def _post(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> tuple[Any, int]:
    started = time.perf_counter()
    last_err = ""
    for attempt in range(4):
        response = requests.post(url, headers=headers, json=body, timeout=timeout)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        try:
            payload = response.json()
        except ValueError:
            payload = {"_raw": response.text[:500]}
        if response.ok:
            return payload, elapsed_ms
        last_err = f"HTTP {response.status_code} {url}: {str(payload)[:400]}"
        if response.status_code == 429 and attempt < 3:
            time.sleep(8 * (attempt + 1))
            continue
        raise RuntimeError(last_err)
    raise RuntimeError(last_err)


def _get(url: str, headers: dict[str, str], params: dict[str, Any], timeout: int) -> tuple[Any, int]:
    started = time.perf_counter()
    response = requests.get(url, headers=headers, params=params, timeout=timeout)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    try:
        payload = response.json()
    except ValueError:
        payload = {"_raw": response.text[:500]}
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code} {url}: {str(payload)[:400]}")
    return payload, elapsed_ms


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _norm_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def _clip(text: str | None, limit: int = SNIPPET_CHARS) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _hit(url: str, title: str | None, snippet: str | None) -> dict[str, Any]:
    return {
        "url": url,
        "title": (title or "").strip(),
        "snippet": _clip(snippet),
        "host": _host(url),
    }


def _dedupe(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for hit in hits:
        key = _norm_url(hit["url"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out[:MAX_RESULTS]


def search_exa(question: str, _case: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    body, elapsed_ms = _post(
        "https://api.exa.ai/search",
        {"x-api-key": os.environ["EXA_API_KEY"], "Content-Type": "application/json"},
        {"query": question, "type": "auto", "numResults": MAX_RESULTS, "contents": {"highlights": True}},
        timeout=45,
    )
    hits = []
    for item in body.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        highlights = item.get("highlights") or []
        snippet = " ".join(h for h in highlights if isinstance(h, str)) or item.get("text")
        hits.append(_hit(item["url"], item.get("title"), snippet))
    return _dedupe(hits), elapsed_ms


def search_firecrawl(question: str, _case: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    body, elapsed_ms = _post(
        "https://api.firecrawl.dev/v2/search",
        {"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}", "Content-Type": "application/json"},
        {"query": question, "limit": MAX_RESULTS},
        timeout=60,
    )
    data = body.get("data") if isinstance(body, dict) else body
    if isinstance(data, dict):
        rows = data.get("web") or data.get("results") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    hits = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not url:
            continue
        hits.append(_hit(url, item.get("title"), item.get("description") or item.get("snippet")))
    return _dedupe(hits), elapsed_ms


def search_parallel(question: str, _case: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    body, elapsed_ms = _post(
        "https://api.parallel.ai/v1/search",
        {"x-api-key": os.environ["PARALLEL_API_KEY"], "Content-Type": "application/json"},
        {
            "objective": question,
            "search_queries": [question],
            "mode": "turbo",
            "advanced_settings": {"max_results": MAX_RESULTS},
        },
        timeout=60,
    )
    hits = []
    for item in body.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        excerpts = item.get("excerpts") or []
        hits.append(_hit(item["url"], item.get("title"), "\n".join(e for e in excerpts if isinstance(e, str))))
    return _dedupe(hits), elapsed_ms


def search_predictleads(question: str, case: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    domain = case.get("company_domain") or (case.get("gold") or {}).get("domain")
    if not domain:
        raise RuntimeError("PredictLeads needs company_domain")
    recipe = case.get("recipe") or case.get("pattern")
    categories = expand_categories(recipe_categories(recipe), recipe)
    try:
        picked, _, _ = pick_categories(_openai(), "gpt-4.1-mini", question, recipe)
        if picked:
            categories = picked
    except Exception:  # noqa: BLE001
        pass
    hits, raw = fetch_category_events(domain, categories)
    return hits[:MAX_RESULTS], int(raw.get("latency_ms") or 0)


SEARCHERS = {
    "exa": search_exa,
    "firecrawl": search_firecrawl,
    "parallel": search_parallel,
    "predictleads": search_predictleads,
}


def _openai() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _json_completion(client: OpenAI, model: str, system: str, user: str) -> tuple[dict[str, Any], dict[str, int]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    usage = response.usage
    tokens = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    text = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}, tokens
    return (data if isinstance(data, dict) else {}), tokens


def _openai_usd(tokens: dict[str, int]) -> float:
    return (
        tokens.get("input_tokens", 0) / 1_000_000 * UNIT_COST_USD["openai_input_per_m"]
        + tokens.get("output_tokens", 0) / 1_000_000 * UNIT_COST_USD["openai_output_per_m"]
    )


def extract_answer(client: OpenAI, model: str, question: str, hits: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    packed = [{"url": h["url"], "title": h["title"], "snippet": h["snippet"]} for h in hits]
    data, tokens = _json_completion(
        client,
        model,
        "Answer only from the provided search snippets. No extra research. "
        "If the snippets do not contain the fact, answer empty. "
        "Return JSON {\"answer\": string, \"cited_url\": string}.",
        json.dumps({"question": question, "results": packed}, ensure_ascii=False),
    )
    return {"answer": str(data.get("answer") or "").strip(), "cited_url": str(data.get("cited_url") or "").strip()}, tokens


def score_gold(client: OpenAI, model: str, question: str, expected: str, cells: list[dict[str, Any]], answer: str) -> tuple[dict[str, Any], dict[str, int]]:
    if not (answer or "").strip():
        return {"correct": False, "cells_hit": 0, "note": "empty answer"}, {"input_tokens": 0, "output_tokens": 0}
    data, tokens = _json_completion(
        client,
        model,
        "Score against gold. correct=true only if every gold cell is present. "
        "Return JSON {\"correct\": bool, \"cells_hit\": int, \"note\": string}.",
        json.dumps({"question": question, "expected_answer": expected, "cells": cells, "answer": answer}, ensure_ascii=False),
    )
    return {
        "correct": bool(data.get("correct")),
        "cells_hit": int(data.get("cells_hit") or 0),
        "note": str(data.get("note") or ""),
    }, tokens


def source_metrics(hits: list[dict[str, Any]], gold_url: str) -> dict[str, Any]:
    gold = _norm_url(gold_url)
    gold_host = _host(gold_url)
    gold_tail = Path(urlparse(gold_url).path).name.lower()
    rank = None
    for i, hit in enumerate(hits, 1):
        key = _norm_url(hit["url"])
        if key == gold or (gold_host == hit["host"] and gold_tail and gold_tail in (urlparse(hit["url"]).path or "").lower()):
            rank = i
            break
    return {
        "gold_rank": rank,
        "recall_at_5": rank is not None and rank <= 5,
        "recall_at_10": rank is not None and rank <= 10,
        "mrr": round(1 / rank, 3) if rank else 0.0,
    }


def answer_in_excerpt(hits: list[dict[str, Any]], cells: list[dict[str, Any]]) -> bool:
    blob = " ".join(f"{h.get('title')} {h.get('snippet')}" for h in hits).lower()
    for cell in cells:
        value = str(cell.get("value") or "").lower().strip()
        if not value:
            continue
        token = re.sub(r"[^a-z0-9.$]+", " ", value).strip()
        if token and token in blob:
            return True
        digits = re.sub(r"[^\d]", "", value)
        if len(digits) >= 2 and digits in re.sub(r"[^\d]", "", blob):
            return True
    return False


def run_case(client: OpenAI, model: str, case: dict[str, Any], vendors: list[str]) -> dict[str, Any]:
    question = case["question"]
    gold_url = (case.get("gold") or {}).get("primary_url") or ""
    cells = case.get("cells") or (case.get("gold") or {}).get("cells") or []
    expected = case.get("expected_answer") or ""

    def _one(name: str) -> tuple[str, list[dict[str, Any]], int, str | None]:
        try:
            hits, elapsed_ms = SEARCHERS[name](question, case)
            return name, hits, elapsed_ms, None
        except Exception as exc:  # noqa: BLE001
            return name, [], 0, str(exc)

    per_hits: dict[str, list[dict[str, Any]]] = {}
    per_meta: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(vendors)) as pool:
        for future in as_completed([pool.submit(_one, name) for name in vendors]):
            name, hits, elapsed_ms, error = future.result()
            per_hits[name] = hits
            per_meta[name] = {"latency_ms": elapsed_ms, "error": error}

    vendors_out: dict[str, Any] = {}
    gold_scores: dict[str, Any] = {}
    cost_events: list[dict[str, Any]] = []
    for name in vendors:
        hits = per_hits[name]
        extracted = {"answer": "", "cited_url": ""}
        extract_tokens = {"input_tokens": 0, "output_tokens": 0}
        if not per_meta[name]["error"]:
            extracted, extract_tokens = extract_answer(client, model, question, hits)
            cost_events.append({"kind": "vendor_search", "vendor": name, "usd": UNIT_COST_USD[name], "ok": True})
        else:
            # Failed vendor calls (402/5xx) are still usually billed except auth/credit refusals.
            billed = "402" not in str(per_meta[name]["error"])
            cost_events.append({"kind": "vendor_search", "vendor": name, "usd": UNIT_COST_USD[name] if billed else 0.0, "ok": False})
        gold, score_tokens = score_gold(client, model, question, expected, cells, extracted["answer"])
        gold_scores[name] = gold
        llm_usd = _openai_usd(extract_tokens) + _openai_usd(score_tokens)
        cost_events.append(
            {
                "kind": "openai",
                "vendor": name,
                "usd": round(llm_usd, 6),
                "input_tokens": extract_tokens["input_tokens"] + score_tokens["input_tokens"],
                "output_tokens": extract_tokens["output_tokens"] + score_tokens["output_tokens"],
            }
        )
        vendors_out[name] = {
            **per_meta[name],
            "hits": hits,
            **extracted,
            "gold": gold,
            "source": source_metrics(hits, gold_url),
            "answer_in_excerpt": answer_in_excerpt(hits, cells),
        }
    return {
        "id": case.get("id"),
        "domain": "company_news",
        "pattern": case.get("recipe") or case.get("pattern"),
        "company": case.get("company"),
        "company_domain": case.get("company_domain"),
        "question": question,
        "expected_answer": expected,
        "search_queries": [question],
        "vendors": vendors_out,
        "gold_scores": gold_scores,
        "cost_events": cost_events,
    }


def print_case(row: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"{row.get('id')}  [company_news/{row.get('pattern')}]")
    print(row["question"])
    for name, payload in row["vendors"].items():
        src = payload.get("source") or {}
        mark = "Y" if (payload.get("gold") or {}).get("correct") else "n"
        excerpt = "Y" if payload.get("answer_in_excerpt") else "n"
        if payload.get("error"):
            print(f"  {name}  ERROR {payload['error'][:160]}")
            continue
        print(
            f"  {name}  {payload.get('latency_ms')}ms  hits={len(payload.get('hits') or [])}  "
            f"gold={mark} excerpt={excerpt} r@5={src.get('recall_at_5')} mrr={src.get('mrr')}  "
            f"{(payload.get('answer') or '')[:140]}"
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vendors = sorted({n for r in rows for n in (r.get("gold_scores") or {})})
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_recipe.setdefault(row.get("pattern") or "?", []).append(row)

    def _block(items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(items), "vendors": {}}
        for name in vendors:
            correct = [bool(((r.get("gold_scores") or {}).get(name) or {}).get("correct")) for r in items]
            r5 = [bool(((r.get("vendors") or {}).get(name) or {}).get("source", {}).get("recall_at_5")) for r in items]
            r10 = [bool(((r.get("vendors") or {}).get(name) or {}).get("source", {}).get("recall_at_10")) for r in items]
            mrr = [float(((r.get("vendors") or {}).get(name) or {}).get("source", {}).get("mrr") or 0) for r in items]
            excerpt = [bool(((r.get("vendors") or {}).get(name) or {}).get("answer_in_excerpt")) for r in items]
            out["vendors"][name] = {
                "accuracy": round(sum(correct) / len(correct), 3) if correct else None,
                "recall_at_5": round(sum(r5) / len(r5), 3) if r5 else None,
                "recall_at_10": round(sum(r10) / len(r10), 3) if r10 else None,
                "mrr": round(sum(mrr) / len(mrr), 3) if mrr else None,
                "answer_in_excerpt": round(sum(excerpt) / len(excerpt), 3) if excerpt else None,
            }
        split_n = 0
        for row in items:
            flags = [bool(((row.get("gold_scores") or {}).get(n) or {}).get("correct")) for n in vendors]
            if flags and not all(flags) and any(flags):
                split_n += 1
        out["split_n"] = split_n
        out["split_rate"] = round(split_n / len(items), 3) if items else None
        return out

    recipes = {name: _block(items) for name, items in sorted(by_recipe.items())}
    ranked = sorted(recipes.items(), key=lambda kv: (kv[1]["split_n"], kv[1]["split_rate"] or 0), reverse=True)
    return {"overall": _block(rows), "recipes_ranked_by_splits": [{"recipe": k, **v} for k, v in ranked]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--vendors", default=",".join(VENDORS))
    parser.add_argument("--model", default=os.getenv("WEBSEARCH_PROBE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated sample ids to run")
    args = parser.parse_args()

    load_environment()
    vendors = [v.strip() for v in args.vendors.split(",") if v.strip()]
    cases = json.loads(args.samples.read_text())
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c.get("id") in wanted]
    if args.limit:
        cases = cases[: args.limit]
    print(f"model={args.model} vendors={vendors} cases={len(cases)} protocol=single-query max_results={MAX_RESULTS}")

    client = _openai()
    rows: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "data" / "websearch-probe" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "run.partial.json"

    def _cost_rollups() -> dict[str, Any]:
        events = [e for row in rows for e in (row.get("cost_events") or [])]
        by_kind: dict[str, dict[str, Any]] = {}
        for event in events:
            key = event["vendor"] if event["kind"] == "vendor_search" else "openai"
            bucket = by_kind.setdefault(key, {"calls": 0, "usd": 0.0})
            if key == "openai":
                bucket.setdefault("input_tokens", 0)
                bucket.setdefault("output_tokens", 0)
                bucket["input_tokens"] += int(event.get("input_tokens") or 0)
                bucket["output_tokens"] += int(event.get("output_tokens") or 0)
            bucket["calls"] += 1
            bucket["usd"] = round(bucket["usd"] + float(event.get("usd") or 0), 6)
        return {
            "unit_cost_usd": UNIT_COST_USD,
            "by_endpoint": by_kind,
            "total_usd": round(sum(v["usd"] for v in by_kind.values()), 4),
        }

    def _write(path: Path, summary: dict[str, Any] | None) -> None:
        path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "protocol": "atomic_single_query",
                    "model": args.model,
                    "vendors": vendors,
                    "summary": summary,
                    "cost": _cost_rollups(),
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )

    for i, case in enumerate(cases):
        if i:
            time.sleep(1.0)
        row = run_case(client, args.model, case, vendors)
        rows.append(row)
        print_case(row)
        sys.stdout.flush()
        if (i + 1) % 4 == 0 or i + 1 == len(cases):
            _write(partial, summarize(rows))

    summary = summarize(rows)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    cost = _cost_rollups()
    print("COST")
    print(json.dumps(cost, indent=2))
    out_path = out_dir / "run.json"
    _write(out_path, summary)
    (out_dir / "cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
