"""Offline SearchPipe request and search-only result contracts."""

import os
import unittest
from unittest.mock import patch

from websearch import run_official_news_probe as news
from websearch import score_official_search_metrics as score


class SearchPipeContracts(unittest.TestCase):
    def test_searchpipe_arms(self):
        payload = {"results": [
            {"url": "https://example.com/news", "title": "News", "content": "Snippet",
             "raw_content": "Full page", "evidence": {"quote": "Evidence"}},
        ], "answer": "Generated answer"}
        for arm, depth, cost in (
            ("searchpipe_basic", "basic", 0.01),
            ("searchpipe_advanced", "advanced", 0.02),
        ):
            with self.subTest(arm=arm), patch.dict(os.environ, {"SEARCHPIPE_API_KEY": "test-key"}):
                with patch.object(news, "_http", return_value={"ok": True, "response": payload}) as call:
                    hits, _ = news.call_endpoint(arm, "Unchanged question?", {})
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.kwargs["url"], "https://searchpipe.tech/search")
                self.assertEqual(call.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
                self.assertEqual(call.call_args.kwargs["body"], {
                    "query": "Unchanged question?", "search_depth": depth, "max_results": 10,
                    "include_answer": False, "include_raw_content": False,
                })
                self.assertEqual(hits[0]["snippet"], "Snippet")
                self.assertNotIn("Full page", str(hits))
                self.assertNotIn("Generated answer", str(hits))
                self.assertEqual(news.UNIT_COST_USD[arm], cost)
                self.assertEqual(news._parse_endpoints(arm), [arm])
                self.assertIn(arm, score.ENDPOINTS)


if __name__ == "__main__":
    unittest.main()
