"""Offline SpecStream request contract tests; no credentials or API calls required."""

import os
import unittest
from unittest.mock import patch

from websearch import run_official_news_probe as runner
from websearch.score_official_search_metrics import ENDPOINTS as SCORED_ENDPOINTS


class SpecStreamTests(unittest.TestCase):
    def test_request_is_unmodified_search_and_sources_are_normalized(self):
        response = {
            "kind": "fact",
            "answer": {"value": "Must not leak"},
            "rows": [{"name": "Row", "page": "https://example.com/row", "snippet": "Row text"}],
            "sources": [
                {"url": "https://example.com/news", "title": "News", "snippet": "Search evidence"},
                {"url": "https://example.com/news", "snippet": "Duplicate"},
                None,
                {"title": "Missing URL"},
                *[{"url": f"https://example.com/{i}", "snippet": str(i)} for i in range(12)],
            ],
        }
        with patch.dict(os.environ, {"SPECSTREAM_API_KEY": "test"}), patch.object(
            runner, "_http", return_value={"ok": True, "response": response}
        ) as http:
            hits, _ = runner.call_endpoint("specstream", "Unmodified question?", {})
        kwargs = http.call_args.kwargs
        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api.specstream.ai/v1/web-search")
        self.assertEqual(kwargs["params"], {"q": "Unmodified question?"})
        self.assertIsNone(kwargs["body"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test")
        self.assertEqual(hits[0]["snippet"], "Search evidence")
        self.assertEqual(len({h["url"] for h in hits}), len(hits))
        self.assertNotIn("Must not leak", str(hits))
        self.assertIn("specstream", runner.ENDPOINTS)
        self.assertIn("specstream", SCORED_ENDPOINTS)
        self.assertEqual(runner.ENDPOINT_ENV["specstream"], ("SPECSTREAM_API_KEY",))

    def test_failed_request_returns_no_hits(self):
        with patch.dict(os.environ, {"SPECSTREAM_API_KEY": "test"}), patch.object(
            runner, "_http", return_value={"ok": False, "status_code": 402, "response": {}}
        ):
            hits, raw = runner.call_endpoint("specstream", "Question", {})
        self.assertEqual(hits, [])
        self.assertEqual(raw["status_code"], 402)

    def test_bad_payloads_give_no_hits(self):
        self.assertEqual(runner._parse_specstream(None), [])
        self.assertEqual(runner._parse_specstream({"sources": "x"}), [])


if __name__ == "__main__":
    unittest.main()
