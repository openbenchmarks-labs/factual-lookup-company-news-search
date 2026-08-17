# web search benchmark — raw audit trail

Each `(question × endpoint)` cell on the public leaderboard has two files
under `company-news-public-119/`:

- `<endpoint>.json` — slim cell used to recompute the headline numbers
- `<endpoint>.raw.json` — the same cell plus the redacted HTTP envelope and
  the extract / accuracy / AR judge prompts

The literal vendor bytes also live at
`data/company-news/official-runs/20260816T020806Z/raw_calls/<case>/<endpoint>.json`.
