"""Offline Keenable request-contract tests; no credentials or API calls required."""

import os
import unittest
from unittest.mock import MagicMock, patch

from websearch import run_official_news_probe as runner
from websearch.score_official_search_metrics import ENDPOINTS as SCORED_ENDPOINTS

ARM = "keenable"


class KeenableTests(unittest.TestCase):
    def test_arms_send_search_contract_and_normalize_results(self):
        response = {"query": "q", "mode": "pro", "results": [
            {"url": "https://example.com/news", "title": "News", "description": "", "snippet": "Search evidence"},
            {"url": "https://example.com/news", "snippet": "Duplicate"},
            None, {"title": "Missing URL", "snippet": "x"}, {"url": "  ", "snippet": "blank"},
            *[{"url": f"https://example.com/{i}", "snippet": str(i)} for i in range(12)],
        ]}
        with patch.dict(os.environ, {"KEENABLE_API_KEY": "test-key"}), patch.object(
            runner, "_http", return_value={"ok": True, "response": response}
        ) as http:
            hits, _ = runner.call_endpoint(ARM, "Unmodified question?", {})
        self.assertEqual(http.call_count, 1)
        kwargs = http.call_args.kwargs
        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["url"], "https://api.keenable.ai/v1/search")
        self.assertEqual(kwargs["headers"]["X-API-Key"], "test-key")
        self.assertEqual(kwargs["body"], {"query": "Unmodified question?", "mode": "pro", "max_results": 10})
        self.assertEqual(len(hits), 10)
        self.assertEqual(hits[0]["snippet"], "Search evidence")
        self.assertEqual(len({h["url"] for h in hits}), 10)

    def test_roster_membership(self):
        self.assertIn(ARM, runner.ENDPOINTS)
        self.assertIn(ARM, SCORED_ENDPOINTS)
        self.assertIn("mode=pro", runner.ENDPOINT_MAP[ARM])
        self.assertEqual(runner.ENDPOINT_ENV[ARM], ("KEENABLE_API_KEY",))
        self.assertEqual(runner._parse_endpoints(ARM), [ARM])
        self.assertEqual(runner._parse_endpoints("")[0], ARM)
        # Exactly one Keenable arm in every roster and price table.
        for table in (runner.ENDPOINTS, SCORED_ENDPOINTS, runner.UNIT_COST_USD, runner.ENDPOINT_MAP, runner.ENDPOINT_ENV):
            self.assertEqual([k for k in table if k.startswith("keenable")], [ARM])

    def test_parser_falls_back_to_description_and_tolerates_empty_payloads(self):
        hits = runner._parse_keenable({"results": [{"url": "https://example.com", "description": "Desc only"}]})
        self.assertEqual(hits[0]["snippet"], "Desc only")
        odd = runner._parse_keenable({"results": [
            {"url": "https://example.com/a", "title": 5, "snippet": {"x": 1}, "description": "Fallback", "published_at": 1767225600},
            {"url": "https://example.com/b", "title": None, "snippet": "   ", "description": None},
        ]})
        self.assertEqual([(h["title"], h["snippet"]) for h in odd], [("", "Fallback"), ("", "")])
        for payload in (None, {}, {"results": None}, {"results": "bad"}, []):
            self.assertEqual(runner._parse_keenable(payload), [])

    def test_failed_request_does_not_extract_error_body(self):
        with patch.dict(os.environ, {"KEENABLE_API_KEY": "test-key"}), patch.object(
            runner, "_http", return_value={"ok": False, "status_code": 401, "response": {"results": [
                {"url": "https://example.com", "snippet": "x"}]}}
        ):
            hits, raw = runner.call_endpoint(ARM, "Question", {})
        self.assertEqual(hits, [])
        self.assertEqual(raw["status_code"], 401)

    def test_cost_uses_reported_credits_else_list_price(self):
        self.assertEqual(runner.UNIT_COST_USD[ARM], 0.004)
        with_credits = {"ok": True, "response": {"results": [], "usage": {"sku": "search", "credits": 2}}}
        self.assertAlmostEqual(runner._vendor_search_usd(ARM, with_credits), 0.008)
        for response in ({"results": []}, {"usage": {"credits": None}}, None):
            self.assertEqual(runner._vendor_search_usd(ARM, {"ok": True, "response": response}), 0.004)
        # Other vendors stay on list price even if a payload carries usage.
        self.assertEqual(runner._vendor_search_usd("exa_fast", with_credits), runner.UNIT_COST_USD["exa_fast"])

    def test_auth_failure_is_not_billed(self):
        fetched = (
            [], {"ok": False, "status_code": 401, "response": {}, "latency_ms": 5, "attempts": [], "error": "HTTP 401"},
        )
        with patch.object(runner, "call_endpoint", return_value=fetched), \
                patch.object(runner, "_write_raw", return_value="raw.json"), \
                patch.object(runner, "score_gold", return_value=({"correct": False}, {"input_tokens": 0, "output_tokens": 0})):
            row = runner.run_case(MagicMock(), "m", {"id": "c1", "question": "Q"}, [ARM], MagicMock())
        search = [e for e in row["cost_events"] if e["kind"] == "vendor_search"]
        self.assertEqual(search[0]["usd"], 0.0)

    def test_api_key_header_is_redacted(self):
        redacted = runner._redact_headers({"X-API-Key": "secret", "x-api-key": "secret", "Content-Type": "application/json"})
        self.assertEqual(redacted["X-API-Key"], "***REDACTED***")
        self.assertEqual(redacted["x-api-key"], "***REDACTED***")
        self.assertEqual(redacted["Content-Type"], "application/json")


if __name__ == "__main__":
    unittest.main()
