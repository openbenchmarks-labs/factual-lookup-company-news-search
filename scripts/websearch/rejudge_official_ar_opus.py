#!/usr/bin/env python3
"""Score AR@K with Claude Opus on Bedrock instead of string/digit heuristics.

One judge call per question × endpoint. The model returns which of the top-10
snippets already contain every gold cell. AR@1 / AR@5 / AR@10 are derived from
those ranks. Reuses stored hits. No vendor API calls. No Anthropic API key.

  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_ar_opus.py
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_ar_opus.py --run
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_ar_opus.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from websearch.bedrock_judge import (  # noqa: E402
    call_bedrock_json,
    judge_model_id,
    load_bedrock_environment,
)

RUN_DIR = ROOT / "data" / "company-news" / "official-runs" / "20260816T020806Z"
RUN_PATH = RUN_DIR / "run.json"
SAMPLES = ROOT / "data" / "company-news" / "samples.json"
MANIFEST = ROOT / "data" / "company-news" / "manifest.json"
CKPT_DIR = RUN_DIR / "rejudge-ar-opus5"

JUDGE_LABEL = "claude-opus-5"

AR_PROMPT = """Decide which search snippets already contain the gold fact.

A snippet is answer-bearing only if a reader could recover every gold cell from
that snippet alone, without outside knowledge. Synonyms and equivalent numeric
forms are allowed (eleven million = $11 million = $11M). Digit coincidence is
not enough: do not treat "2.0" as matching "2026", or "$20M" as matching
"$11 million". Stale directory pages that name a different person, company, or
round are not answer-bearing.

Ranks are 1-indexed in the order the snippets are listed.

### RESPONSE FORMAT (CRITICAL)

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:

{
  "bearing_ranks": [1, 4],
  "note": "short reason"
}

bearing_ranks is the list of ranks whose snippet contains every gold cell.
Use [] if none do. note is at most 12 words.
"""


class ArOut(BaseModel):
    bearing_ranks: list[int] = Field(default_factory=list)
    note: str = ""

    @field_validator("bearing_ranks", mode="before")
    @classmethod
    def _ranks(cls, value: Any) -> list[int]:
        if not isinstance(value, list):
            return []
        out: list[int] = []
        for item in value:
            try:
                rank = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= rank <= 10:
                out.append(rank)
        return sorted(set(out))


class Job(BaseModel):
    case_id: str
    endpoint: str
    question: str
    expected: str
    cells: list[dict[str, Any]]
    snippets: list[dict[str, Any]]


def _ckpt_path(case_id: str, endpoint: str) -> Path:
    return CKPT_DIR / f"{case_id}__{endpoint}.json"


def _jobs() -> list[Job]:
    payload = json.loads(RUN_PATH.read_text())
    samples = {row["id"]: row for row in json.loads(SAMPLES.read_text())}
    scored = set(json.loads(MANIFEST.read_text())["ids"])
    jobs: list[Job] = []
    for case in payload.get("cases") or []:
        case_id = case["id"]
        if case_id not in scored:
            continue
        sample = samples.get(case_id)
        if not sample:
            continue
        cells = sample.get("cells") or (sample.get("gold") or {}).get("cells") or []
        expected = sample.get("expected_answer") or sample.get("ground_truth") or ""
        for endpoint, block in (case.get("vendors") or {}).items():
            hits = list(block.get("hits") or [])[:10]
            jobs.append(
                Job(
                    case_id=case_id,
                    endpoint=endpoint,
                    question=case["question"],
                    expected=expected,
                    cells=cells,
                    snippets=[
                        {
                            "rank": index,
                            "title": (hit.get("title") or "")[:300],
                            "url": hit.get("url") or "",
                            "snippet": (hit.get("snippet") or "")[:1200],
                        }
                        for index, hit in enumerate(hits, 1)
                    ],
                )
            )
    return jobs


def _score(job: Job) -> tuple[ArOut, dict[str, int]]:
    parsed, tokens = call_bedrock_json(
        AR_PROMPT,
        json.dumps(
            {
                "question": job.question,
                "expected_answer": job.expected,
                "cells": job.cells,
                "snippets": job.snippets,
            },
            ensure_ascii=False,
        ),
        temperature=0.0,
        max_tokens=2048,
    )
    try:
        out = ArOut.model_validate(parsed)
    except ValidationError as exc:
        raise RuntimeError(f"judge JSON failed validation: {parsed!r}") from exc
    n = len(job.snippets)
    out.bearing_ranks = [rank for rank in out.bearing_ranks if rank <= n]
    return out, tokens


def _run_one(job: Job) -> dict[str, Any]:
    dest = _ckpt_path(job.case_id, job.endpoint)
    if dest.exists():
        return json.loads(dest.read_text())
    if not job.snippets:
        ar = ArOut(bearing_ranks=[], note="no snippets")
        score_tokens = {"input_tokens": 0, "output_tokens": 0}
    else:
        ar, score_tokens = _score(job)
    row = {
        "case_id": job.case_id,
        "endpoint": job.endpoint,
        "bearing_ranks": ar.bearing_ranks,
        "ar1": any(rank <= 1 for rank in ar.bearing_ranks),
        "ar5": any(rank <= 5 for rank in ar.bearing_ranks),
        "ar10": any(rank <= 10 for rank in ar.bearing_ranks),
        "note": ar.note,
        "n_snippets": len(job.snippets),
        "judge_model": JUDGE_LABEL,
        "judge_model_id": judge_model_id(),
        "judge_transport": "bedrock-converse",
        "score_tokens": score_tokens,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2) + "\n")
    return row


def _apply() -> dict[str, Any]:
    payload = json.loads(RUN_PATH.read_text())
    applied = 0
    missing = 0
    for case in payload.get("cases") or []:
        for endpoint, block in (case.get("vendors") or {}).items():
            path = _ckpt_path(case["id"], endpoint)
            if not path.exists():
                missing += 1
                continue
            row = json.loads(path.read_text())
            ranks = [int(r) for r in (row.get("bearing_ranks") or []) if int(r) >= 1]
            block["ar"] = {
                "bearing_ranks": ranks,
                "ar1": any(rank <= 1 for rank in ranks),
                "ar5": any(rank <= 5 for rank in ranks),
                "ar10": any(rank <= 10 for rank in ranks),
                "note": row.get("note") or "",
                "judge_model": row.get("judge_model") or JUDGE_LABEL,
                "judge_model_id": row.get("judge_model_id"),
                "judge_transport": row.get("judge_transport"),
                "judged_at": row.get("judged_at"),
            }
            block["answer_in_excerpt"] = any(rank <= 10 for rank in ranks)
            applied += 1
    payload["ar_judge_model"] = JUDGE_LABEL
    payload["ar_judge_model_id"] = judge_model_id()
    payload["ar_judge_transport"] = "bedrock-converse"
    payload["ar_rejudged_at"] = datetime.now(timezone.utc).isoformat()
    RUN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return {"applied": applied, "missing": missing}


def _has_bedrock_creds() -> bool:
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        return True
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True
    if (os.getenv("AWS_PROFILE") or "").strip():
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Call Bedrock. Default prints the plan.")
    parser.add_argument("--apply", action="store_true", help="Write checkpoints back onto run.json.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    load_bedrock_environment()
    jobs = _jobs()
    done = sum(1 for job in jobs if _ckpt_path(job.case_id, job.endpoint).exists())
    pending = [job for job in jobs if not _ckpt_path(job.case_id, job.endpoint).exists()]
    if args.limit:
        pending = pending[: args.limit]
    plan = {
        "run": str(RUN_PATH),
        "scored_jobs": len(jobs),
        "checkpointed": done,
        "pending": len(pending),
        "with_snippets_pending": sum(1 for job in pending if job.snippets),
        "judge_model": JUDGE_LABEL,
        "judge_model_id": judge_model_id(),
        "judge_transport": "bedrock-converse",
        "vendor_calls": 0,
        "anthropic_api": False,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        if not _has_bedrock_creds():
            raise RuntimeError(
                "Bedrock credentials missing. Set AWS_BEARER_TOKEN_BEDROCK or "
                "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (no ANTHROPIC_API_KEY)."
            )
        ok = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                    ok += 1
                    print(
                        json.dumps(
                            {
                                "ok": True,
                                "n": ok,
                                "case": job.case_id,
                                "endpoint": job.endpoint,
                                "bearing_ranks": row.get("bearing_ranks"),
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(
                        json.dumps(
                            {
                                "ok": False,
                                "case": job.case_id,
                                "endpoint": job.endpoint,
                                "error": str(exc)[:400],
                            }
                        ),
                        flush=True,
                    )
        print(json.dumps({"finished": True, "ok": ok, "failed": failed}), flush=True)
        if failed:
            return 1
    if args.apply:
        print(json.dumps(_apply(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
