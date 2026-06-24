import unittest

from vibeharness.config import Config
from vibeharness.memory import NarrativeMemory
from vibeharness.prompt import SystemPromptBuilder, build_turn_prompt
from vibeharness.toolset import default_catalog


class NarrativeMemoryTest(unittest.TestCase):
    def test_empty(self):
        self.assertIn("not taken any actions", NarrativeMemory().render())

    def test_first_then_connectors(self):
        m = NarrativeMemory()
        m.record("you wrote a file")
        m.record("you read it back")
        rendered = m.render()
        self.assertEqual(rendered, "First, you wrote a file\nThen, you read it back")
        self.assertEqual(len(m), 2)


class PromptTest(unittest.TestCase):
    def setUp(self):
        catalog = default_catalog()
        self.registry = catalog.build_registry(catalog.select(["fs"]), Config())

    def test_system_prompt_contains_tools_and_schema(self):
        sp = SystemPromptBuilder(self.registry).build()
        self.assertIn("write_file", sp)
        self.assertIn("validate", sp)
        self.assertIn("Action schema", sp)
        self.assertIn("oneOf", sp)

    def test_turn_prompt_contains_task_and_narrative(self):
        prompt = build_turn_prompt("make a file", "First, you did a thing")
        self.assertIn("make a file", prompt)
        self.assertIn("First, you did a thing", prompt)
        self.assertIn("next action", prompt.lower())


if __name__ == "__main__":
    unittest.main()
