#!/usr/bin/env python3
"""Re-extract and re-score the official company-news run. No vendor API calls.

Uses stored hits only.
  extract harness: gpt-5.6-terra @ medium
  gold judge:      gpt-5.6-sol @ medium

  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_extract_and_gold.py
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_extract_and_gold.py --run
  PYTHONPATH=scripts .venv/bin/python -u scripts/websearch/rejudge_official_extract_and_gold.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from _shared import load_environment  # noqa: E402

RUN_DIR = ROOT / "data" / "company-news" / "official-runs" / "20260816T020806Z"
RUN_PATH = RUN_DIR / "run.json"
SAMPLES = ROOT / "data" / "company-news" / "samples.json"
MANIFEST = ROOT / "data" / "company-news" / "manifest.json"
CKPT_DIR = RUN_DIR / "rejudge-gpt56"

EXTRACT_MODEL = "gpt-5.6-terra"
JUDGE_MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"

EXTRACT_PROMPT = (
    "Answer only from the provided search snippets. No extra research. "
    "If the snippets do not contain the fact, answer empty. "
    "Do not use outside knowledge."
)
SCORE_PROMPT = (
    "Score the extracted answer against gold. correct=true only if every gold "
    "cell is present in the answer. Synonyms and equivalent numeric forms are "
    "allowed. Do not use outside knowledge or the search snippets."
)


class ExtractOut(BaseModel):
    answer: str = ""
    cited_url: str = ""


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
    hits: list[dict[str, Any]]
    had_error: bool = False


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=180)


def _parse(client: OpenAI, model: str, system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, dict[str, int]]:
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": REASONING_EFFORT},
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text_format=schema,
                max_output_tokens=4000,
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("empty output_parsed")
            usage = getattr(response, "usage", None)
            tokens = {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
            return parsed, tokens
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{model} failed after 3 attempts: {last}")


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
                    hits=hits,
                    had_error=bool(block.get("error")),
                )
            )
    return jobs


def _run_one(job: Job) -> dict[str, Any]:
    dest = _ckpt_path(job.case_id, job.endpoint)
    if dest.exists():
        return json.loads(dest.read_text())
    packed = [{"url": h.get("url"), "title": h.get("title"), "snippet": h.get("snippet")} for h in job.hits]
    extract_tokens = {"input_tokens": 0, "output_tokens": 0}
    if not packed or job.had_error:
        extracted = ExtractOut()
    else:
        client = _client()
        extracted, extract_tokens = _parse(
            client,
            EXTRACT_MODEL,
            EXTRACT_PROMPT,
            json.dumps({"question": job.question, "results": packed}, ensure_ascii=False),
            ExtractOut,
        )
    answer = (extracted.answer or "").strip()
    score_tokens = {"input_tokens": 0, "output_tokens": 0}
    if not answer:
        gold = GoldOut(correct=False, cells_hit=0, note="empty answer")
    else:
        client = _client()
        gold, score_tokens = _parse(
            client,
            JUDGE_MODEL,
            SCORE_PROMPT,
            json.dumps(
                {
                    "question": job.question,
                    "expected_answer": job.expected,
                    "cells": job.cells,
                    "answer": answer,
                },
                ensure_ascii=False,
            ),
            GoldOut,
        )
    row = {
        "case_id": job.case_id,
        "endpoint": job.endpoint,
        "answer": answer,
        "cited_url": (extracted.cited_url or "").strip(),
        "gold": gold.model_dump(),
        "extract_model": EXTRACT_MODEL,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "extract_tokens": extract_tokens,
        "score_tokens": score_tokens,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2) + "\n")
    return row


def _apply() -> dict[str, Any]:
    payload = json.loads(RUN_PATH.read_text())
    backup = RUN_DIR / "run.gpt-4.1-mini.json"
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
            block["answer"] = row.get("answer") or ""
            block["cited_url"] = row.get("cited_url") or ""
            block["gold"] = row.get("gold") or {"correct": False, "cells_hit": 0, "note": ""}
            block["rejudge"] = {
                "extract_model": row.get("extract_model"),
                "judge_model": row.get("judge_model"),
                "reasoning_effort": row.get("reasoning_effort"),
                "judged_at": row.get("judged_at"),
            }
            applied += 1
            if before != after:
                flipped += 1
    payload["model"] = EXTRACT_MODEL
    payload["judge_model"] = JUDGE_MODEL
    payload["judge_reasoning_effort"] = REASONING_EFFORT
    payload["rejudged_at"] = datetime.now(timezone.utc).isoformat()
    RUN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return {"applied": applied, "missing": missing, "flipped": flipped, "backup": str(backup)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Call OpenAI. Default prints the plan.")
    parser.add_argument("--apply", action="store_true", help="Write checkpoints back onto run.json.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    load_environment()
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
        "extract_model": EXTRACT_MODEL,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "vendor_calls": 0,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required")
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
                                "error": str(exc)[:300],
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
