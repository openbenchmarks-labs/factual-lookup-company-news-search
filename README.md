# OpenBenchmarks Web Search Benchmark

Open head-to-head leaderboard for web search APIs.

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

**Live benchmark:** https://openbenchmarks.com/web-search

This repo is the open data + code mirror of that page — every cell on the
leaderboard is backed by a literal HTTP request/response envelope (auth
headers redacted) and the extract + judge artifacts under `data/websearch-runs/`.

Each question is sent **as-is** to every endpoint: one request, max 10
results, title + snippet only, no query rewrite and no page fetch. Extracts
use `gpt-5.6-terra` (medium reasoning). Accuracy and AR@K are judged by
`claude-opus-5` on Amazon Bedrock. The frozen public snapshot is
**company-news-public-119**: 119 company-news questions × 12 endpoints.

## Endpoints

- **Live benchmark UI** — https://openbenchmarks.com/web-search
- **JSON API** — https://openbenchmarks.com/api/benchmarks/web-search
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json

## Current leaderboard

| # | Vendor | Accuracy | AR@5 | Mean latency | List price / query |
|---|---|---|---|---|---|
| 1 | Exa deep | 99.16% | 99.20% | 2954 ms | $0.012 |
| 2 | Exa instant | 99.16% | 97.50% | 444 ms | $0.007 |
| 3 | Parallel advanced | 98.32% | 98.30% | 3327 ms | $0.005 |
| 4 | Linkup standard | 97.48% | 95.00% | 2343 ms | $0.005 |
| 5 | Parallel basic | 97.48% | 93.30% | 1401 ms | $0.005 |
| 6 | Linkup fast | 96.64% | 96.60% | 1843 ms | $0.005 |
| 7 | Brave Search | 94.96% | 95.00% | 697 ms | $0.005 |
| 8 | Tavily advanced | 94.96% | 95.00% | 4345 ms | $0.016 |
| 9 | Firecrawl | 94.12% | 93.30% | 1104 ms | $0.005 |
| 10 | SERP (RapidAPI) | 94.12% | 93.30% | 2136 ms | $0.003 |
| 11 | Seltz news | 68.91% | 64.70% | 321 ms | $0.005 |
| 12 | PredictLeads news events | 63.03% | 63.00% | 668 ms | $0.040 |

119 questions × 12 endpoints. Parallel turbo, Tavily ultra-fast, and
Seltz companies are **not** in this public snapshot. PredictLeads and Seltz
are specialized news indexes, not general web search; they are included so
those two surfaces can be compared.

The full per-cell breakdown and the raw audit trail live under
`data/websearch-runs/` and `data/company-news/official-runs/20260816T020806Z/raw_calls/`.

## What's in this repo

| path | purpose |
|---|---|
| `data/latest-websearch.json` | Leaderboard snapshot — 119 questions, per-vendor metrics, ranked rows. |
| `data/company-news/samples.json` | Frozen questions + gold cells + source URLs. |
| `data/company-news/ground_truth.json` | Compact gold records. |
| `data/company-news/dropped-flagged-13.json` | The 13 labeller-flagged items excluded from scoring. |
| `data/company-news/official-runs/20260816T020806Z/run.json` | Filtered official run (public endpoints, n=119). |
| `data/company-news/official-runs/20260816T020806Z/raw_calls/<case>/<endpoint>.json` | Literal vendor HTTP request/response; auth headers redacted. |
| `data/websearch-runs/company-news-public-119/<case>/<endpoint>.json` | Slim cell: hits, extract, accuracy verdict, AR@K verdict. |
| `data/websearch-runs/company-news-public-119/<case>/<endpoint>.raw.json` | Slim cell plus HTTP envelope and reconstructed judge prompts. |
| `manifest.json` | Flat index of every cell with headline numbers + file paths. |
| `scripts/run_websearch_benchmark.py` | Vendor sweep orchestrator (public endpoints by default). |
| `scripts/websearch/run_official_news_probe.py` | HTTP adapters, redaction, resume/attach. |
| `scripts/websearch/rejudge_official_extract_and_gold.py` | Terra extract from stored hits (no vendor calls). |
| `scripts/websearch/rejudge_official_gold_opus.py` | Opus accuracy judge (Bedrock converse). |
| `scripts/websearch/rejudge_official_ar_opus.py` | Opus AR@K judge on snippets (Bedrock converse). |
| `scripts/websearch/score_official_search_metrics.py` | Recompute accuracy / AR@K / latency from the snapshot. |
| `scripts/verify_public_artifacts.py` | Zero-network integrity check of the frozen 119 × 12 snapshot. |

See [DATA.md](DATA.md) for schemas.

## Reproducing a cell

Pick any (question, endpoint) pair. Then:

- Replay **`http`** in `data/websearch-runs/company-news-public-119/<case>/<endpoint>.raw.json` (or the file under `raw_calls/`) with your own credentials.
- Re-extract from the stored `hits` with `gpt-5.6-terra`.
- Re-score accuracy and AR@K by sending `judge_calls[].system` + `judge_calls[].user` to any judge; the published snapshot uses `claude-opus-5` via Bedrock.

## Running the benchmark yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env && $EDITOR .env

PYTHONPATH=scripts python scripts/verify_public_artifacts.py
PYTHONPATH=scripts python scripts/run_websearch_benchmark.py --limit 1 --endpoints exa_instant
```

A full live sweep spends vendor credits. Existing official-run cells are not
deleted; pass `--resume` / `--attach` as documented in the runner.

After a live sweep, re-extract and re-judge:

```bash
PYTHONPATH=scripts python scripts/websearch/rejudge_official_extract_and_gold.py --run --apply
PYTHONPATH=scripts python scripts/websearch/rejudge_official_gold_opus.py --run --apply
PYTHONPATH=scripts python scripts/websearch/rejudge_official_ar_opus.py --run --apply
PYTHONPATH=scripts python scripts/websearch/score_official_search_metrics.py
```

## Methodology

- **Protocol.** One natural-language question, one HTTP request, `max 10`
  results. No rewrite, no second query, no fetching the linked pages. The
  extract model sees title + snippet only.
- **Gold.** Harvested independently and locked before scoring. 13 labeller-flagged
  items were dropped; scored **n=119**.
- **Extract.** `gpt-5.6-terra` at medium reasoning. Empty extract scores
  incorrect for accuracy.
- **Accuracy.** `claude-opus-5` marks `correct=true` only when every gold cell
  is present in the extract. Synonyms and equivalent numeric forms are allowed.
  The judge does not read the snippets.
- **AR@K.** `claude-opus-5` labels which of the top-10 snippets already contain
  every gold cell. AR@1 / AR@5 / AR@10 are derived from those ranks. This is
  not string/digit matching.
- **PredictLeads.** `GET /api/v3/companies/{domain}/news_events` — the
  question text is not sent. Tagged as a company-news events index.
- **Seltz.** `POST /v1/search` with `scope=news`. The API does not return a
  title field.
- **Cost.** Published list price per request as of 16 Aug 2026. Free tiers and
  volume discounts are not included.
- **Known limitations.** Extract can miss a fact that is already in a snippet
  (accuracy 0, AR@5 1). Conflicting later hits can also make the extract
  abstain. Every raw call and judge note is published so either can be audited.

No vendor sponsors or controls this benchmark.
