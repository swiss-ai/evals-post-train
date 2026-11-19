"""bfcl_request: the thinking settings reach the requests bfcl-eval's OpenAI handler makes."""

import unittest

from scripts.evalscope import bfcl_request


class RequestSettingsTests(unittest.TestCase):
    def test_only_reasoning_effort_and_extra_body_are_sent(self):
        config = {"temperature": 0.6, "top_p": 0.95, "reasoning_effort": "high", "extra_body": {"chat_template_kwargs": {"x": 1}}}
        self.assertEqual(bfcl_request.request_settings(config), {"reasoning_effort": "high", "extra_body": {"chat_template_kwargs": {"x": 1}}})

    def test_unset_settings_are_left_out(self):
        self.assertEqual(bfcl_request.request_settings({"temperature": 0.0, "reasoning_effort": None}), {})


class InstallTests(unittest.TestCase):
    def setUp(self):
        try:
            from bfcl_eval.model_handler.api_inference import openai_completion
        except ImportError:
            self.skipTest("bfcl_eval is not installed")
        self.handler = openai_completion.OpenAICompletionsHandler
        self._saved = self.handler.generate_with_backoff
        self.handler._bfcl_request_installed = False
        self.calls = []
        self.handler.generate_with_backoff = lambda handler, **kwargs: self.calls.append(kwargs)

    def tearDown(self):
        self.handler.generate_with_backoff = self._saved
        self.handler._bfcl_request_installed = False

    def test_requests_carry_the_settings_and_keep_the_handlers_own(self):
        bfcl_request.install({"reasoning_effort": "high", "extra_body": {"a": 1}})
        self.handler.generate_with_backoff(object(), messages=[], model="m", temperature=0.6)
        self.assertEqual(self.calls, [{"messages": [], "model": "m", "temperature": 0.6, "reasoning_effort": "high", "extra_body": {"a": 1}}])

    def test_what_the_handler_sets_wins_and_a_second_install_changes_nothing(self):
        bfcl_request.install({"reasoning_effort": "high"})
        first = self.handler.generate_with_backoff
        bfcl_request.install({"reasoning_effort": "low"})
        self.assertIs(self.handler.generate_with_backoff, first)
        self.handler.generate_with_backoff(object(), reasoning_effort="medium")
        self.assertEqual(self.calls[0]["reasoning_effort"], "medium")

    def test_nothing_to_send_leaves_the_handler_alone(self):
        before = self.handler.generate_with_backoff
        bfcl_request.install({"temperature": 0.0})
        self.assertIs(self.handler.generate_with_backoff, before)


if __name__ == "__main__":
    unittest.main()
