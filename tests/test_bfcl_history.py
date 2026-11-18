import importlib.util
import json
import unittest

from scripts.evalscope import bfcl_history

HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_BFCL = importlib.util.find_spec("bfcl_eval") is not None

# What vLLM sends for the fields a reply does not use: explicit nulls, which the SDK keeps on the object.
WIRE = {
    "role": "assistant", "content": None, "refusal": None, "annotations": None, "audio": None,
    "function_call": None, "tool_calls": None, "reasoning": None,
}
CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


class CleanMessageTests(unittest.TestCase):
    def test_empty_reply_becomes_an_empty_text_turn(self):
        self.assertEqual(bfcl_history.clean_message(WIRE), {"role": "assistant", "content": ""})

    def test_text_reply_keeps_its_content_and_loses_the_nulls(self):
        self.assertEqual(bfcl_history.clean_message({**WIRE, "content": "hi"}), {"role": "assistant", "content": "hi"})

    def test_empty_tool_call_list_counts_as_no_tool_calls(self):
        self.assertEqual(bfcl_history.clean_message({**WIRE, "content": "hi", "tool_calls": []}), {"role": "assistant", "content": "hi"})

    def test_tool_call_turn_keeps_its_calls_and_gets_no_content_key(self):
        self.assertEqual(bfcl_history.clean_message({**WIRE, "tool_calls": [CALL]}), {"role": "assistant", "tool_calls": [CALL]})

    def test_tool_call_turn_keeps_content_when_there_is_some(self):
        cleaned = bfcl_history.clean_message({**WIRE, "content": "calling", "tool_calls": [CALL]})
        self.assertEqual(cleaned, {"role": "assistant", "content": "calling", "tool_calls": [CALL]})

    @unittest.skipUnless(HAS_OPENAI, "openai not installed (requirements/evalscope-runtime.lock)")
    def test_sdk_message_object_with_explicit_nulls(self):
        from openai.types.chat import ChatCompletionMessage

        message = ChatCompletionMessage.model_validate({**WIRE, "tool_calls": [CALL]})
        self.assertEqual(bfcl_history.clean_message(message)["tool_calls"][0]["id"], "c1")
        self.assertNotIn("refusal", bfcl_history.clean_message(message))


@unittest.skipUnless(HAS_OPENAI and HAS_BFCL, "openai and bfcl-eval not installed (requirements/evalscope-runtime.lock)")
class InstallTests(unittest.TestCase):
    def test_history_sent_back_has_no_nulls_in_either_mode(self):
        import httpx
        import openai
        from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
        from openai.types.chat import ChatCompletionMessage

        bfcl_history.install()
        bfcl_history.install()  # a second call must not wrap again
        sent = {}

        def respond(request):
            sent["messages"] = json.loads(request.content)["messages"]
            return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})

        client = openai.OpenAI(api_key="k", base_url="http://x/v1", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        handler = OpenAICompletionsHandler.__new__(OpenAICompletionsHandler)
        for add in (handler._add_assistant_message_FC, handler._add_assistant_message_prompting):
            inference_data = {"message": [{"role": "user", "content": "hi"}]}
            reply = ChatCompletionMessage.model_validate(WIRE)
            add(inference_data, {"model_responses_message_for_chat_history": reply})
            client.chat.completions.create(model="m", messages=inference_data["message"])
            self.assertEqual(sent["messages"][1], {"role": "assistant", "content": ""})


if __name__ == "__main__":
    unittest.main()
