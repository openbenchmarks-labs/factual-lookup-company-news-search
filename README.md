# Web Search API Benchmark for AI Agents: Factual Lookup on Company News

Open, independent benchmark of web search APIs for AI agents on factual lookup:
TinyFish, Parallel, Perplexity, Linkup, Firecrawl, Brave Search, You, Exa,
Tavily, and a Google SERP API. 300 company-news questions, one query each, no
query rewrite and no page fetch, scored on extracted-answer accuracy and ranked
by cost per 1,000 correct answers. Open source code + open data.

**Live leaderboard:** https://openbenchmarks.com/company-news
**Public dataset:** [`openbenchmarks/OB-News-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-News-Websearch)

Published by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

This repo is **open code only**. It does not include run dumps, raw vendor
HTTP, or leaderboard snapshots.

## Which web search API is cheapest for AI agents?

Ranked by **$ / 1k correct**: the list price of 1,000 queries divided by
extracted-answer accuracy, which is what you pay for 1,000 right answers.
Accuracy alone ignores the bill and raw cost per query ignores misses, so
cost-to-quality is the number that compounds when an agent fires this call all
day.

**TinyFish** is free at 92.0% accuracy, capped at 30 requests/min and 500/hour,
so $0 is not unlimited throughput. The best-value paid row is **Parallel
(mode=fast)** at $1.16 per 1,000 correct.

| # | Vendor | Endpoint | $ / 1k correct | Accuracy | AR@1 | AR@5 | Latency | Snippet tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | TinyFish | GET api.search.tinyfish.ai | Free (30 req/min) | 92.0% | 74.3% | 90.7% | 2.62s | 441 |
| 2 | Parallel fast | POST /v1/search mode=fast | $1.16 | 86.0% | 44.3% | 79.0% | 942ms | 1,839 |
| 3 | Parallel turbo | POST /v1/search mode=turbo | $1.40 | 71.3% | 45.3% | 66.0% | 348ms | 1,853 |
| 4 | SERP (RapidAPI) | GET google-search74 limit=10 | $3.13 | 96.0% | 78.0% | 95.0% | 751ms | 497 |
| 5 | Perplexity (low) | POST /search context=low | $5.14 | 97.3% | 91.7% | 98.3% | 1.38s | 476 |
| 6 | Linkup fast | POST /v1/search depth=fast | $5.17 | 96.7% | 79.0% | 94.7% | 1.57s | 3,022 |
| 7 | Firecrawl | POST /v2/search | $5.24 | 95.3% | 77.7% | 96.7% | 510ms | 678 |
| 8 | Brave LLM Context | POST /res/v1/llm/context | $5.32 | 94.0% | 81.0% | 94.7% | 601ms | 2,064 |
| 9 | You | POST /v1/search count=10 | $5.36 | 93.3% | 84.7% | 94.7% | 532ms | 2,708 |
| 10 | Parallel basic | POST /v1/search mode=basic | $5.36 | 93.3% | 55.0% | 92.0% | 1.68s | 2,330 |
| 11 | Brave Search | GET /res/v1/web/search | $5.36 | 93.3% | 79.3% | 91.7% | 630ms | 817 |
| 12 | Linkup standard | POST /v1/search depth=standard | $5.43 | 92.0% | 67.3% | 90.3% | 2.55s | 2,983 |
| 13 | You highlights | POST /v1/search highlights | $5.51 | 90.7% | 72.0% | 89.7% | 628ms | 2,837 |
| 14 | Exa fast | POST /search type=fast | $7.05 | 99.3% | 95.0% | 99.3% | 652ms | 1,987 |
| 15 | Exa instant | POST /search type=instant | $7.17 | 97.7% | 80.0% | 97.3% | 398ms | 2,128 |
| 16 | Tavily ultra-fast | POST /search depth=ultra-fast | $60.02 | 13.3% | 10.0% | 17.0% | 191ms | 2,827 |

Cost per 1,000 queries is the published PAYG list price, not promotional packs
or volume discounts. The
[live board](https://openbenchmarks.com/company-news) is the source of truth;
re-read it before quoting these numbers.

Full ranking: https://openbenchmarks.com/web-search/cheapest-search-api

## Which web search API is most accurate for AI agents?

**Exa (type=fast)** leads at 99.3% extracted-answer accuracy, followed by
Perplexity (low) at 97.3% and Exa instant at 97.7%. Exa fast also leads answer
recall, with the correct answer already in the first snippet 95.0% of the time.

Accuracy is tightly bunched: eleven of sixteen rows land between 90% and 99%.
On a one-query lookup the interesting spread is cost and latency, not accuracy.
The exception is Tavily ultra-fast at 13.3%, which trades essentially all of its
accuracy for the fastest response on the board.

Full ranking: https://openbenchmarks.com/web-search/most-accurate-search-api

## Which web search API is fastest for AI agents?

**Tavily ultra-fast** returns in 191ms, but at 13.3% accuracy it is the wrong
row for almost any agent. The useful fast rows are **Parallel turbo** at 348ms
and **Exa instant** at 398ms, which hold 71.3% and 97.7% accuracy respectively.

Latency is mean wall time of the search request only, excluding the extract and
judge calls.

Full ranking: https://openbenchmarks.com/web-search/fastest-search-api

## Which web search API is most token-efficient?

Measured as extracted-answer accuracy per 1,000 snippet tokens, which is what
matters when the snippets land in an agent's context window on every call.
**TinyFish** leads at roughly 209 accuracy points per 1,000 snippet tokens,
ahead of Perplexity (low) at 204 and SERP at 193.

Snippet volume varies by nearly 7x across the board, from 441 tokens for
TinyFish to 3,022 for Linkup fast, with no corresponding accuracy gain.

Full ranking: https://openbenchmarks.com/web-search/most-token-efficient-search-api

## How it is scored

Each question is sent **as-is**: one request, max 10 results, title + snippet
only, no query rewrite and no page fetch. Extracts use `gpt-5.6-terra`
(medium reasoning). Accuracy and AR@K are judged by `claude-opus-5` on
Amazon Bedrock, an independent model family from the extractor, and the judge
never fetches the live page.

- **Accuracy.** The extractor writes an answer from the returned titles and
  snippets; the judge scores that answer against human-labelled ground truth.
- **AR@1 and AR@5.** Answer recall, meaning whether the ground truth was already
  present in the first snippet, or in any of the top five, with no extract step.
- **Latency.** Mean wall time of the search request.
- **$ / 1k correct.** Cost per 1,000 queries divided by extracted-answer
  accuracy. The ranking metric.

The model-only baseline is 0: the questions use company facts dated after the
model's training cutoff, so a correct answer has to be found rather than
recalled.

**Datasets.** The live board is scored on a locked **300-question** private
set. A **100-question** public set (question + ground truth; source URLs
omitted) is on Hugging Face as
[`openbenchmarks/OB-News-Websearch`](https://huggingface.co/datasets/openbenchmarks/OB-News-Websearch).
Use the public rows to inspect the format and run this harness; scores on
those rows are not comparable to the board.

## What is here

| path | purpose |
|---|---|
| `scripts/run_websearch_benchmark.py` | Vendor sweep orchestrator |
| `scripts/websearch/run_official_news_probe.py` | HTTP adapters, redaction, resume/attach |
| `scripts/websearch/rejudge_official_extract_and_gold.py` | Terra extract from stored hits |
| `scripts/websearch/rejudge_official_gold_opus.py` | Opus accuracy judge (Bedrock converse) |
| `scripts/websearch/rejudge_official_ar_opus.py` | Opus AR@K judge on snippets |
| `scripts/websearch/score_official_search_metrics.py` | Recompute accuracy / AR@K / latency |
| `scripts/websearch/bedrock_judge.py` | Bedrock converse helper |

## Public endpoints

These are the general web-search arms on the live factual-lookup board:

| CLI id | Call |
|---|---|
| `tinyfish` | GET api.search.tinyfish.ai |
| `parallel_fast` | POST /v1/search mode=fast |
| `parallel_turbo` | POST /v1/search mode=turbo |
| `serp` | GET google-search74.p.rapidapi.com limit=10 |
| `perplexity_low` | POST /search search_context_size=low |
| `linkup_fast` | POST /v1/search depth=fast |
| `firecrawl` | POST /v2/search |
| `brave_llm` | POST /res/v1/llm/context |
| `you` | POST /v1/search |
| `parallel_basic` | POST /v1/search mode=basic |
| `brave` | GET /res/v1/web/search |
| `linkup_standard` | POST /v1/search depth=standard |
| `you_highlights` | POST /v1/search extraction_mode=highlights |
| `exa_fast` | POST /search type=fast |
| `exa_instant` | POST /search type=instant |
| `tavily_basic` | POST /search search_depth=basic |
| `tavily_advanced` | POST /search search_depth=advanced |
| `datahyena` | GET /v1/companies/timeline |

`--backend` / `--endpoints` default to this list. TinyFish Search is free
with a 30 req/min cap. The runner can carry arms that are not yet on the
published board.

The live page also bands dedicated news indexes (Seltz news, Autobound,
PredictLeads, Datahyena) on the same 300 questions. Of those, `datahyena` has
an adapter here; the rest are not in this runner.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env

PYTHONPATH=scripts python scripts/run_websearch_benchmark.py --limit 1 --endpoints tinyfish
```

A full live sweep spends vendor credits. After a sweep, re-extract and re-judge:

```bash
PYTHONPATH=scripts python scripts/websearch/rejudge_official_extract_and_gold.py --run --apply
PYTHONPATH=scripts python scripts/websearch/rejudge_official_gold_opus.py --run --apply
PYTHONPATH=scripts python scripts/websearch/rejudge_official_ar_opus.py --run --apply
PYTHONPATH=scripts python scripts/websearch/score_official_search_metrics.py
```

## Related benchmarks

This is the factual-lookup task of the OpenBenchmarks web search benchmark. The
same vendors are measured on two other jobs:

- **Hard retrieval.** Coding-agent tickets against enterprise docs, scored on
  grounded task completion: https://openbenchmarks.com/web-search-for-coding-agents
- **Multi-hop search.** Multi-constraint company discovery, scored on F1:
  https://openbenchmarks.com/multi-turn-company-search
- **Methodology and all three boards:** https://openbenchmarks.com/web-search
- **Agent-readable index:** https://openbenchmarks.com/llms.txt

## License

MIT.
