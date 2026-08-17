#!/usr/bin/env python3
"""Verify the frozen public-119 web-search artifacts. No network calls."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/latest-websearch.json"
SAMPLES = ROOT / "data/company-news/samples.json"
MANIFEST = ROOT / "manifest.json"
RUN = ROOT / "data/company-news/official-runs/20260816T020806Z/run.json"
ENDPOINTS = {
    "exa_instant", "exa_deep", "parallel_basic", "parallel_advanced",
    "firecrawl", "predictleads", "seltz_news", "brave", "tavily_advanced",
    "serp", "linkup_fast", "linkup_standard",
}
HIDDEN = {"parallel_turbo", "tavily_ultrafast", "seltz_companies"}


def main() -> int:
    snapshot = json.loads(SNAPSHOT.read_text())
    samples = json.loads(SAMPLES.read_text())
    manifest = json.loads(MANIFEST.read_text())
    run = json.loads(RUN.read_text())
    assert snapshot["dataset_slug"] == "company-news-public-119"
    assert snapshot["n"] == 119
    assert len(samples) == 119
    assert len(snapshot["cases"]) == 119
    assert len(run["cases"]) == 119
    assert set(snapshot["endpoints"]) == ENDPOINTS
    assert not (set(run["endpoints"]) & HIDDEN)
    cells = manifest["cells"]
    assert len(cells) == 119 * 12
    assert {cell["endpoint"] for cell in cells} == ENDPOINTS
    missing = []
    for cell in cells:
        for key in ("slim_file", "raw_file", "http_file"):
            path = ROOT / cell[key]
            if not path.exists():
                missing.append(str(path))
        slim = json.loads((ROOT / cell["slim_file"]).read_text())
        raw = json.loads((ROOT / cell["raw_file"]).read_text())
        assert slim["endpoint"] == cell["endpoint"]
        assert "http" in raw
        assert len(raw.get("judge_calls") or []) == 3
        vendors = next(case["vendors"] for case in run["cases"] if case["id"] == cell["case_slug"])
        assert cell["endpoint"] in vendors
        assert not (set(vendors) & HIDDEN)
    assert not missing, missing[:10]
    for row in snapshot["leaderboard"]:
        subset = [c for c in cells if c["endpoint"] == row["endpoint"]]
        assert len(subset) == 119
        assert sum(c["accuracy"] for c in subset) == row["correct"]
    print("cases: 119")
    print("endpoints: 12")
    print("cells: 1428")
    print("artifact verification passed; network calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
