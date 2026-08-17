#!/usr/bin/env python3
"""Re-score official company-news extracts with Claude Opus on Bedrock.

Reuses stored terra extracts. No vendor API calls. No Anthropic API key.

  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_gold_opus.py
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_gold_opus.py --run
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_gold_opus.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

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
CKPT_DIR = RUN_DIR / "rejudge-opus5"

EXTRACT_MODEL = "gpt-5.6-terra"
JUDGE_LABEL = "claude-opus-5"

SCORE_PROMPT = """Score the extracted answer against gold. correct=true only if every gold cell is present in the answer. Synonyms and equivalent numeric forms are allowed. Do not use outside knowledge or the search snippets.

### RESPONSE FORMAT (CRITICAL)

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:

{
  "correct": true,
  "cells_hit": 0,
  "note": "short reason"
}

correct must be a boolean. cells_hit is how many gold cells are present in the answer. note is a short string.
"""


class GoldOut(BaseModel):
    correct: bool
    cells_hit: int = 0
    note: str = ""


class Job(BaseModel):
    case_id: str
    endpoint: str
    question: str
    expected: str
    cells: list[dict[str, Any]]
    answer: str
    cited_url: str = ""
    extract_model: str = EXTRACT_MODEL


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
            rejudge = block.get("rejudge") or {}
            jobs.append(
                Job(
                    case_id=case_id,
                    endpoint=endpoint,
                    question=case["question"],
                    expected=expected,
                    cells=cells,
                    answer=(block.get("answer") or "").strip(),
                    cited_url=(block.get("cited_url") or "").strip(),
                    extract_model=rejudge.get("extract_model") or payload.get("model") or EXTRACT_MODEL,
                )
            )
    return jobs


def _score(job: Job) -> tuple[GoldOut, dict[str, int]]:
    parsed, tokens = call_bedrock_json(
        SCORE_PROMPT,
        json.dumps(
            {
                "question": job.question,
                "expected_answer": job.expected,
                "cells": job.cells,
                "answer": job.answer,
            },
            ensure_ascii=False,
        ),
        temperature=0.0,
        max_tokens=1024,
    )
    try:
        return GoldOut.model_validate(parsed), tokens
    except ValidationError as exc:
        raise RuntimeError(f"judge JSON failed validation: {parsed!r}") from exc


def _run_one(job: Job) -> dict[str, Any]:
    dest = _ckpt_path(job.case_id, job.endpoint)
    if dest.exists():
        return json.loads(dest.read_text())
    if not job.answer:
        gold = GoldOut(correct=False, cells_hit=0, note="empty answer")
        score_tokens = {"input_tokens": 0, "output_tokens": 0}
    else:
        gold, score_tokens = _score(job)
    row = {
        "case_id": job.case_id,
        "endpoint": job.endpoint,
        "answer": job.answer,
        "cited_url": job.cited_url,
        "gold": gold.model_dump(),
        "extract_model": job.extract_model,
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
    backup = RUN_DIR / "run.gpt-5.6-sol.json"
    if not backup.exists():
        shutil.copy2(RUN_PATH, backup)
    flipped = 0
    applied = 0
    missing = 0
    for case in payload.get("cases") or []:
        for endpoint, block in (case.get("vendors") or {}).items():
            path = _ckpt_path(case["id"], endpoint)
            if not path.exists():
                missing += 1
                continue
            row = json.loads(path.read_text())
            before = bool((block.get("gold") or {}).get("correct"))
            after = bool((row.get("gold") or {}).get("correct"))
            block["gold"] = row.get("gold") or {"correct": False, "cells_hit": 0, "note": ""}
            block["rejudge"] = {
                "extract_model": row.get("extract_model") or EXTRACT_MODEL,
                "judge_model": row.get("judge_model") or JUDGE_LABEL,
                "judge_model_id": row.get("judge_model_id"),
                "judge_transport": row.get("judge_transport"),
                "judged_at": row.get("judged_at"),
            }
            applied += 1
            if before != after:
                flipped += 1
    payload["model"] = EXTRACT_MODEL
    payload["judge_model"] = JUDGE_LABEL
    payload["judge_model_id"] = judge_model_id()
    payload["judge_transport"] = "bedrock-converse"
    payload.pop("judge_reasoning_effort", None)
    payload["rejudged_at"] = datetime.now(timezone.utc).isoformat()
    RUN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return {"applied": applied, "missing": missing, "flipped": flipped, "backup": str(backup)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Call Bedrock. Default prints the plan.")
    parser.add_argument("--apply", action="store_true", help="Write checkpoints back onto run.json.")
    parser.add_argument("--workers", type=int, default=4)
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
        "nonempty_pending": sum(1 for job in pending if job.answer),
        "extract_model": EXTRACT_MODEL,
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
                                "correct": (row.get("gold") or {}).get("correct"),
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


def _has_bedrock_creds() -> bool:
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        return True
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True
    if (os.getenv("AWS_PROFILE") or "").strip():
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
