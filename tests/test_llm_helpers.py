import unittest

from vibeharness.llm import OllamaClient


class LLMHelperTest(unittest.TestCase):
    def test_render_chatml_structure(self):
        prompt = OllamaClient._render_chatml("SYS", "USER")
        self.assertIn("<|im_start|>system\nSYS<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nUSER<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_continue_closes_open_think(self):
        out = OllamaClient._continue_after_reasoning("<think>reasoning so far")
        self.assertEqual(out, "<think>reasoning so far</think>\n")

    def test_continue_empty_reasoning(self):
        self.assertEqual(OllamaClient._continue_after_reasoning("   "), "")

    def test_continue_already_closed(self):
        out = OllamaClient._continue_after_reasoning("<think>x</think>answer")
        self.assertEqual(out, "<think>x</think>answer\n")


if __name__ == "__main__":
    unittest.main()
