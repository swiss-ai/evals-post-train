import asyncio
import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

HAS_EVALSCOPE = importlib.util.find_spec("evalscope") is not None


class BasePromptTest(unittest.TestCase):
    def test_stdin_problem_gets_the_one_shot_stdin_example(self):
        from scripts.evalscope import lcb_base

        prompt = lcb_base.base_prompt("Print the sum of two numbers.", None)
        shot = lcb_base.EXAMPLES["stdin"]
        self.assertTrue(prompt.startswith("### Question\n" + shot["question"] + "\n\n### Answer\n\n" + shot["answer"]))
        # the example is followed by the real problem, whose answer is left for the model
        self.assertTrue(prompt.endswith("### Question\nPrint the sum of two numbers.\n\n### Answer\n\n"))
        self.assertNotIn("### Starter Code", prompt)
        self.assertEqual(prompt.count("### Question"), 2)

    def test_starter_code_problem_gets_the_func_example_and_its_starter_code(self):
        from scripts.evalscope import lcb_base

        starter = "class Solution:\n    def f(self, x: int) -> int:\n        "
        prompt = lcb_base.base_prompt("Double x.", starter)
        shot = lcb_base.EXAMPLES["func"]
        self.assertIn("### Starter Code\n" + shot["sample_code"] + "\n\n### Answer\n\n" + shot["answer"], prompt)
        self.assertTrue(prompt.endswith("### Question\nDouble x.\n\n### Starter Code\n" + starter + "\n\n### Answer\n\n"))

    def test_there_is_no_system_message_or_chat_markup(self):
        from scripts.evalscope import lcb_base

        prompt = lcb_base.base_prompt("q", None)
        self.assertNotIn("expert Python programmer", prompt)
        self.assertNotIn("```", prompt.split("### Answer")[-1])

    def test_official_stop_and_token_limit(self):
        from scripts.evalscope import lcb_base

        self.assertEqual((lcb_base.STOP, lcb_base.MAX_TOKENS), (["###"], 2000))


@unittest.skipUnless(HAS_EVALSCOPE, "evalscope not installed (requirements/evalscope-runtime.lock)")
class AdapterPatchTest(unittest.TestCase):
    def setUp(self):
        from scripts.evalscope import lcb_base

        self.lcb_base = lcb_base
        lcb_base.install()
        self.addCleanup(lcb_base.uninstall)

    def test_extraction_keeps_the_reply_as_code(self):
        from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

        adapter = object.__new__(LiveCodeBenchAdapter)
        # the chat extractor would return "" for a reply without a fenced block
        self.assertEqual(adapter.extract_answer("  print(1)\n", None), "print(1)")

    def test_install_is_idempotent_and_reversible(self):
        from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter

        patched = LiveCodeBenchAdapter.extract_answer
        self.lcb_base.install()
        self.assertIs(LiveCodeBenchAdapter.extract_answer, patched)
        self.lcb_base.uninstall()
        adapter = object.__new__(LiveCodeBenchAdapter)
        self.assertEqual(adapter.extract_answer("```python\nprint(2)\n```", None), "print(2)")
        self.lcb_base.install()


@unittest.skipUnless(HAS_EVALSCOPE, "evalscope not installed (requirements/evalscope-runtime.lock)")
class CompletionsBackendTest(unittest.TestCase):
    def _model(self, responses):
        from scripts.evalscope import completions_model

        calls = []

        def create(*, model, prompt, max_tokens=None, stop=None, temperature=None, top_p=None, n=None, extra_body=None,
                   seed=None, timeout=None, frequency_penalty=None, presence_penalty=None, logit_bias=None):
            calls.append(dict(model=model, prompt=prompt, max_tokens=max_tokens, stop=stop, temperature=temperature,
                              top_p=top_p, extra_body=extra_body))
            return responses.pop(0)

        client = SimpleNamespace(completions=SimpleNamespace(create=create))
        model = completions_model.OpenAICompletionsAPI("m", base_url="http://x/v1", api_key="k")
        model.client = client
        return model, calls

    @staticmethod
    def _completion(text="print(1)", finish="stop"):
        return SimpleNamespace(
            model="m", choices=[SimpleNamespace(text=text, finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        )

    @staticmethod
    def _messages():
        from evalscope.api.messages import ChatMessageSystem, ChatMessageUser

        return [ChatMessageSystem(content="You are an expert."), ChatMessageUser(content="### Question\nq\n\n### Answer\n\n")]

    def _config(self, **kw):
        from evalscope.api.model import GenerateConfig

        return GenerateConfig(**kw)

    def test_registered_under_its_eval_type(self):
        from evalscope.api.registry import get_model_api

        from scripts.evalscope import completions_model

        self.assertIs(get_model_api("openai_completions"), completions_model.OpenAICompletionsAPI)

    def test_sends_the_user_text_verbatim_without_the_system_message(self):
        model, calls = self._model([self._completion()])
        out = model.generate(self._messages(), [], "none", self._config(max_tokens=2000, stop_seqs=["###"], temperature=0.6, top_p=0.95))
        self.assertEqual(calls[0]["prompt"], "### Question\nq\n\n### Answer\n\n")
        self.assertEqual((calls[0]["max_tokens"], calls[0]["stop"], calls[0]["temperature"]), (2000, ["###"], 0.6))
        self.assertEqual(out.completion, "print(1)")
        self.assertEqual((out.usage.input_tokens, out.usage.output_tokens, out.usage.total_tokens), (7, 3, 10))

    def test_parameters_the_endpoint_does_not_take_go_in_extra_body(self):
        model, calls = self._model([self._completion()])
        model.generate(self._messages(), [], "none", self._config(max_tokens=10, top_k=20, repetition_penalty=1.1,
                                                                  extra_body={"min_p": 0.05}))
        self.assertEqual(calls[0]["extra_body"], {"min_p": 0.05, "top_k": 20, "repetition_penalty": 1.1})

    def test_length_finish_is_a_max_tokens_stop(self):
        model, _ = self._model([self._completion(finish="length")])
        out = model.generate(self._messages(), [], "none", self._config(max_tokens=5))
        self.assertEqual(out.stop_reason, "max_tokens")

    def test_max_tokens_is_required(self):
        model, _ = self._model([self._completion()])
        with self.assertRaises(ValueError):
            model.generate(self._messages(), [], "none", self._config())

    def test_async_path_gives_the_same_output(self):
        model, calls = self._model([])

        async def create(**kwargs):
            calls.append(kwargs)
            return self._completion("x = 1")

        model._async_client_pool = SimpleNamespace(get=lambda: SimpleNamespace(completions=SimpleNamespace(create=create)))
        out = asyncio.run(model.generate_async(self._messages(), [], "none", self._config(max_tokens=50, stop_seqs=["###"])))
        self.assertEqual(out.completion, "x = 1")
        self.assertEqual(calls[0]["prompt"], "### Question\nq\n\n### Answer\n\n")
        self.assertEqual(calls[0]["stop"], ["###"])
