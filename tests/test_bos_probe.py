import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from scripts.lm_eval_harness.bos_probe import expected_bos, leading_bos, probe


class FakeServer:
    """A /v1/completions endpoint that echoes the given prompt tokens."""

    def __init__(self, tokens, status=200):
        outer = self
        outer.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((body, self.headers.get("Authorization")))
                self.send_response(status)
                self.end_headers()
                payload = {"choices": [{"logprobs": {"tokens": [*tokens, " Paris"]}}]}
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class LeadingBosTests(unittest.TestCase):
    def test_counts_only_leading_tokens(self):
        self.assertEqual(leading_bos(["<s>", "The"], "<s>"), 1)
        self.assertEqual(leading_bos(["<s>", "<s>", "The"], "<s>"), 2)
        self.assertEqual(leading_bos(["The", "<s>"], "<s>"), 0)
        self.assertEqual(leading_bos([], "<s>"), 0)


class ProbeTests(unittest.TestCase):
    def test_one_bos_passes_and_the_prompt_is_the_harness_prompt(self):
        with FakeServer(["<s>", "The", " capital"]) as server:
            ok, message = probe(server.url, "m", "<s>", {"log-likelihood": "<s>The capital"}, api_key="k")
        self.assertTrue(ok, message)
        body, auth = server.requests[0]
        self.assertTrue(body["prompt"].startswith("<s>The capital"))
        self.assertEqual((body["echo"], body["max_tokens"], auth), (True, 1, "Bearer k"))

    def test_no_bos_fails(self):
        with FakeServer(["The", " capital"]) as server:
            ok, message = probe(server.url, "m", "<s>", {"generation": "The capital"})
        self.assertFalse(ok)
        self.assertIn("NO <s>", message)

    def test_no_bos_without_an_expectation_only_warns(self):
        # defines a BOS token but nothing adds it and nothing asked for one (GPT-2 style, a config that lies)
        with FakeServer(["The", " capital"]) as server:
            ok, message = probe(server.url, "m", "<s>", {"generation": "The capital"}, expected=False)
        self.assertTrue(ok)
        self.assertIn("WARNING", message)

    def test_a_double_bos_fails_even_without_an_expectation(self):
        with FakeServer(["<s>", "<s>", "The"]) as server:
            ok, _ = probe(server.url, "m", "<s>", {"generation": "<s>The"}, expected=False)
        self.assertFalse(ok)

    def test_two_bos_after_a_chat_template_only_warns_without_bos_by_client(self):
        # the template's <s> plus the one the server adds: how template runs have always gone
        with FakeServer(["<s>", "<s>", "<|system_start|>"]) as server:
            ok, message = probe(
                server.url, "m", "<s>", {"generation": "<s><|system_start|>"}, expected=True, double_is_error=False
            )
        self.assertTrue(ok)
        self.assertIn("WARNING", message)
        self.assertIn("--bos-by-client", message)

    def test_add_special_tokens_travels_with_the_request_when_given(self):
        with FakeServer(["<s>", "The"]) as server:
            probe(server.url, "m", "<s>", {"generation": "<s>The"}, add_special_tokens=False)
            probe(server.url, "m", "<s>", {"generation": "<s>The"})
        self.assertIs(server.requests[0][0]["add_special_tokens"], False)
        self.assertNotIn("add_special_tokens", server.requests[1][0])

    def test_double_bos_fails(self):
        with FakeServer(["<s>", "<s>", "The"]) as server:
            ok, message = probe(server.url, "m", "<s>", {"generation": "<s>The capital"})
        self.assertFalse(ok)
        self.assertIn("2 <s>", message)

    def test_an_endpoint_without_prompt_tokens_is_skipped_not_failed(self):
        ok, message = probe("http://127.0.0.1:9/v1/completions", "m", "<s>", {"generation": "x"})
        self.assertTrue(ok)
        self.assertIn("skipped", message)


class ExpectedBosTests(unittest.TestCase):
    def test_tokenizer_that_adds_bos_expects_one_whatever_was_asked(self):
        self.assertTrue(expected_bos("<s>", True, False, "auto", True))
        self.assertTrue(expected_bos("<s>", True, False, "false", True))

    def test_a_run_that_prepends_expects_one(self):
        self.assertTrue(expected_bos("<s>", False, True, "true", False))

    def test_auto_that_the_host_lookup_missed_is_still_expected(self):
        # add_bos_token=false and BOS=auto, but the host could not read the config: resolved false
        self.assertTrue(expected_bos("<s>", False, False, "auto", False))

    def test_no_bos_is_respected(self):
        self.assertFalse(expected_bos("<s>", False, False, "false", False))

    def test_a_tokenizer_that_says_nothing_is_not_expected_to_deliver_one(self):
        self.assertFalse(expected_bos("<|endoftext|>", False, False, "auto", None))

    def test_no_bos_token_expects_nothing(self):
        self.assertFalse(expected_bos(None, False, True, "true", False))


if __name__ == "__main__":
    unittest.main()
