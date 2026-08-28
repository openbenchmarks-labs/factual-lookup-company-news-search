#!/usr/bin/env python3
"""Run the public company-news web-search benchmark.

One natural-language query per case, one API request, max 10 results. The
default public sweep includes 14 endpoint configurations, including Parallel
Turbo and Parallel Fast as separately measured arms.
Default extract model is gpt-5.6-terra. Re-score with the Opus judges after
the vendor sweep:

  PYTHONPATH=scripts python scripts/websearch/rejudge_official_extract_and_gold.py --run --apply
  PYTHONPATH=scripts python scripts/websearch/rejudge_official_gold_opus.py --run --apply
  PYTHONPATH=scripts python scripts/websearch/rejudge_official_ar_opus.py --run --apply
  PYTHONPATH=scripts python scripts/websearch/score_official_search_metrics.py
"""
from websearch.run_official_news_probe import main

if __name__ == "__main__":
    main()
