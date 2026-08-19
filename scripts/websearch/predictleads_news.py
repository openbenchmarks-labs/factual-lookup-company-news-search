"""PredictLeads news-events helpers: closed category list, LLM pick, filtered fetch.

The official web-search board does not use this path. It is a product-shaped
adapter: map the question onto PredictLeads categories, filter the company
news_events index, and return those events as extract snippets.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import requests

# Closed set. LLM may only pick from this list.
PREDICTLEADS_CATEGORIES = (
    "acquires",
    "attends_event",
    "closes_offices_in",
    "decreases_headcount_by",
    "expands_facilities",
    "expands_offices_in",
    "expands_offices_to",
    "files_suit_against",
    "goes_public",
    "has_earnings",
    "has_issues_with",
    "has_revenue",
    "has_valuation",
    "hires",
    "identified_as_competitor_of",
    "increases_headcount_by",
    "integrates_with",
    "invests_into",
    "invests_into_assets",
    "is_developing",
    "launches",
    "leaves",
    "opens_new_location",
    "partners_with",
    "promotes",
    "receives_award",
    "receives_financing",
    "recognized_as",
    "retires_from",
    "sells_assets_to",
    "signs_new_client",
    "spins_off_company",
)

# Recipe → categories if the LLM returns nothing usable.
# Funding is tagged from both sides: company-as-recipient vs investor-as-actor.
RECIPE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "receives_financing": ("receives_financing", "invests_into"),
    "financing_lead": ("receives_financing", "invests_into"),
    "acquires": ("acquires", "sells_assets_to"),
    "hires": ("hires", "promotes"),
    "launches": ("launches", "is_developing"),
    "decreases_headcount_by": ("decreases_headcount_by",),
    "expands_offices_to": (
        "expands_offices_to",
        "expands_offices_in",
        "expands_facilities",
        "opens_new_location",
        "launches",
    ),
    "signs_new_client": ("signs_new_client",),
}

# Inverse / sibling tags PredictLeads often uses instead of the obvious label.
CATEGORY_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "receives_financing": ("invests_into", "has_valuation"),
    "invests_into": ("receives_financing",),
    "invests_into_assets": ("invests_into", "receives_financing"),
    "acquires": ("sells_assets_to",),
    "sells_assets_to": ("acquires",),
    "hires": ("promotes",),
    "promotes": ("hires",),
    "launches": ("is_developing",),
    "is_developing": ("launches",),
    "expands_offices_to": ("expands_offices_in", "expands_facilities", "opens_new_location"),
    "expands_offices_in": ("expands_offices_to", "expands_facilities", "opens_new_location"),
    "expands_facilities": ("expands_offices_to", "expands_offices_in", "opens_new_location"),
    "opens_new_location": ("expands_offices_to", "expands_offices_in", "expands_facilities"),
}

# Company news_events has no sort param. Default order is found_at desc
# (when PredictLeads ingested the article). One page of 10 is the API-native
# "latest events" window.
PAGE_SIZE = 10
MAX_EVENTS = 10
MAX_PAGES = 1


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _headers() -> dict[str, str]:
    return {
        "X-Api-Key": os.environ["PREDICT_LEADS_API_KEY"],
        "X-Api-Token": os.environ["PREDICT_LEADS_API_TOKEN"],
        "Accept": "application/json",
    }


def sanitize_categories(raw: Any) -> list[str]:
    allowed = set(PREDICTLEADS_CATEGORIES)
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    for item in items:
        name = str(item or "").strip()
        if name not in allowed or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def recipe_categories(recipe: str | None) -> list[str]:
    return list(RECIPE_CATEGORIES.get(str(recipe or "").strip(), ()))


def expand_categories(picked: list[str], recipe: str | None = None) -> list[str]:
    """Union LLM picks with recipe defaults and inverse/sibling tags."""
    seed = list(picked) + recipe_categories(recipe)
    return sanitize_categories(
        [name for item in seed for name in (item, *CATEGORY_NEIGHBORS.get(item, ()))]
    )


def pick_categories(client: Any, model: str, question: str, recipe: str | None) -> tuple[list[str], dict[str, int], str]:
    """LLM picks every relevant closed-list category, then sibling tags are unioned in."""
    fallback = recipe_categories(recipe)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Pick every PredictLeads news_event category that could contain "
                    "the answer. Use only the provided closed list. Include inverse "
                    "and sibling tags, not just the most precise one. Funding is "
                    "often tagged invests_into (investor side) instead of "
                    "receives_financing (company side); acquisitions may be "
                    "sells_assets_to instead of acquires. Prefer recall: 2-4 "
                    "categories when siblings exist, at most 5. "
                    "Return JSON {\"categories\": string[], \"reason\": string}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "recipe_hint": recipe or "",
                        "allowed_categories": list(PREDICTLEADS_CATEGORIES),
                        "always_include_siblings": {
                            "receives_financing": ["invests_into"],
                            "invests_into": ["receives_financing"],
                            "acquires": ["sells_assets_to"],
                            "hires": ["promotes"],
                            "expands_offices_to": [
                                "expands_offices_in",
                                "expands_facilities",
                                "opens_new_location",
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    usage = response.usage
    tokens = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    try:
        data = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        data = {}
    picked = sanitize_categories((data if isinstance(data, dict) else {}).get("categories"))
    reason = str((data if isinstance(data, dict) else {}).get("reason") or "")
    expanded = expand_categories(picked or fallback, recipe)
    if not expanded:
        return fallback, tokens, reason or "llm empty; used recipe fallback"
    if set(expanded) != set(picked):
        extra = [name for name in expanded if name not in set(picked)]
        reason = (reason + f" Expanded with siblings: {extra}.").strip()
    return expanded, tokens, reason


def _event_to_hit(ev: dict[str, Any], included: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any] | None:
    attrs = ev.get("attributes") or {}
    rel = ((ev.get("relationships") or {}).get("most_relevant_source") or {}).get("data") or {}
    art = included.get(("news_article", rel.get("id") or ""), {})
    url = art.get("url")
    if not url:
        return None
    category = attrs.get("category") or ""
    amount = attrs.get("amount") or ""
    date = attrs.get("effective_date") or ""
    found_at = attrs.get("found_at") or ""
    summary = attrs.get("summary") or ""
    sentence = attrs.get("article_sentence") or art.get("body") or ""
    title = art.get("title") or summary or category
    bits = [b for b in (category, date, amount) if b]
    prefix = f"[{' | '.join(bits)}] " if bits else ""
    snippet = f"{prefix}{summary} {sentence}".strip()
    return {
        "url": url,
        "title": str(title).strip(),
        "snippet": snippet[:2000],
        "host": _host(url),
        "category": category,
        "amount": amount,
        "effective_date": date,
        "found_at": found_at,
        "summary": summary,
    }


def _parse_page(payload: Any) -> list[dict[str, Any]]:
    included: dict[tuple[str, str], dict[str, Any]] = {}
    for inc in (payload or {}).get("included") or []:
        if isinstance(inc, dict) and inc.get("id"):
            included[(inc.get("type") or "", inc["id"])] = inc.get("attributes") or {}
    hits = []
    for ev in (payload or {}).get("data") or []:
        if not isinstance(ev, dict):
            continue
        hit = _event_to_hit(ev, included)
        if hit:
            hits.append(hit)
    return hits


def fetch_category_events(
    domain: str,
    categories: list[str],
    *,
    page_size: int = PAGE_SIZE,
    max_events: int = MAX_EVENTS,
    max_pages: int = MAX_PAGES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """GET company news_events filtered by categories.

    One request, limit=10. The API has no sort=; it returns newest found_at first.
    """
    if not domain:
        return [], {
            "ok": False,
            "error": "PredictLeads needs company_domain",
            "status_code": None,
            "latency_ms": 0,
            "pages": [],
            "categories": categories,
            "url": "",
            "response": None,
        }
    if not categories:
        return [], {
            "ok": False,
            "error": "no PredictLeads categories selected",
            "status_code": None,
            "latency_ms": 0,
            "pages": [],
            "categories": [],
            "url": f"https://predictleads.com/api/v3/companies/{domain}/news_events",
            "response": None,
        }

    url = f"https://predictleads.com/api/v3/companies/{domain}/news_events"
    headers = _headers()
    started = time.perf_counter()
    pages: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_status = None
    last_payload: Any = None
    error = None
    ok = False

    for page in range(1, max_pages + 1):
        params: list[tuple[str, str | int]] = [
            ("limit", page_size),
            ("page", page),
        ]
        params.extend(("categories[]", cat) for cat in categories)
        page_started = time.perf_counter()
        try:
            response = requests.get(url, headers=headers, params=params, timeout=45)
            last_status = response.status_code
            try:
                last_payload = response.json()
            except ValueError:
                last_payload = {"_raw": (response.text or "")[:8000]}
            elapsed_ms = round((time.perf_counter() - page_started) * 1000)
            page_ok = response.ok
            ok = ok or page_ok
            pages.append(
                {
                    "n": page,
                    "status_code": response.status_code,
                    "latency_ms": elapsed_ms,
                    "ok": page_ok,
                    "returned": len((last_payload or {}).get("data") or []) if isinstance(last_payload, dict) else 0,
                }
            )
            if not page_ok:
                error = f"HTTP {response.status_code} {url}: {str(last_payload)[:400]}"
                break
            batch = _parse_page(last_payload)
            if not (last_payload or {}).get("data"):
                break
            for hit in batch:
                key = hit["url"]
                if key in seen:
                    continue
                seen.add(key)
                hits.append(hit)
                if len(hits) >= max_events:
                    break
            if len(hits) >= max_events or len((last_payload or {}).get("data") or []) < page_size:
                break
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            pages.append({"n": page, "ok": False, "error": error})
            break

    return hits[:max_events], {
        "method": "GET",
        "url": url,
        "ok": ok,
        "error": error,
        "status_code": last_status,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "categories": categories,
        "pages": pages,
        "hit_count": len(hits[:max_events]),
        "response": last_payload if len(pages) <= 1 else {"pages": pages, "last": last_payload},
        "request": {
            "headers": {k: "***REDACTED***" for k in _headers()},
            "params": {
                "limit": page_size,
                "page": 1,
                "categories": categories,
                "order": "found_at desc (API default; no sort param)",
            },
        },
    }
