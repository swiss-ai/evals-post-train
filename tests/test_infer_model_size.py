import unittest

from scripts.infer_model_size import infer_parameter_billions


class InferModelSizeTests(unittest.TestCase):
    def test_dense_model_names(self):
        self.assertEqual(infer_parameter_billions("Qwen/Qwen3-8B"), 8)
        self.assertEqual(infer_parameter_billions("Llama-3.2-1.5B-Instruct"), 1.5)
        self.assertEqual(infer_parameter_billions("/models/final_27b/checkpoint"), 27)

    def test_threshold_and_total_parameters(self):
        self.assertEqual(infer_parameter_billions("Qwen3-30B-A3B"), 30)
        self.assertEqual(infer_parameter_billions("Mixtral-8x7B-Instruct"), 56)

    def test_version_numbers_are_not_sizes(self):
        self.assertIsNone(infer_parameter_billions("IFM/K2-Think-V2"))


if __name__ == "__main__":
    unittest.main()
