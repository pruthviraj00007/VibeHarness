import json
import os
import tempfile
import unittest

from vibeharness.agent import RalphAgent
from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.llm import Decision, LLMClient
from vibeharness.registry import ToolRegistry


class FakeLLMClient(LLMClient):
    """Returns scripted actions instead of calling a model. When the script is
    exhausted it repeats the last action (so loops can run to the step budget)."""

    def __init__(self, actions):
        self._actions = actions
        self._i = 0

    def decide(self, system, user, action_schema, on_reason=None, on_action=None):
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        payload = action if isinstance(action, str) else json.dumps(action)
        return Decision(reasoning="", action_json=payload)


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.dir, name)

    def _agent(self, actions, max_steps=10):
        client = FakeLLMClient(actions)
        return RalphAgent(client, self.registry, "SYSTEM", Config(max_steps=max_steps))

    def test_sequential_turns_write_read_finish(self):
        # one action per turn -> three turns
        actions = [
            {"tool": "write_file", "args": {"path": self.p("a.txt"), "content": "hello hello hello"}},
            {"tool": "read_file", "args": {"path": self.p("a.txt")}},
            {"tool": "finish", "args": {"summary": "done"}},
        ]
        result = self._agent(actions).run("make a file")
        self.assertTrue(result.finished)
        self.assertEqual(len(result.turns), 3)
        self.assertEqual(result.final_summary, "done")
        with open(self.p("a.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello hello hello")
        self.assertIn("wrote the file", result.turns[0].actions[0].observation)
        self.assertIn("hello hello hello", result.turns[1].actions[0].observation)

    def test_multiple_actions_in_one_turn(self):
        # a single turn batching three actions (the array form)
        actions = [[
            {"tool": "write_file", "args": {"path": self.p("a.txt"), "content": "batched"}},
            {"tool": "read_file", "args": {"path": self.p("a.txt")}},
            {"tool": "finish", "args": {"summary": "done in one turn"}},
        ]]
        result = self._agent(actions).run("batch it")
        self.assertTrue(result.finished)
        self.assertEqual(len(result.turns), 1)
        self.assertEqual(len(result.turns[0].actions), 3)
        self.assertIn("wrote the file", result.turns[0].actions[0].observation)
        self.assertTrue(result.turns[0].actions[2].final)

    def test_actions_after_finish_are_ignored(self):
        actions = [[
            {"tool": "finish", "args": {"summary": "stop here"}},
            {"tool": "write_file", "args": {"path": self.p("should_not_exist.txt"), "content": "x"}},
        ]]
        result = self._agent(actions).run("t")
        self.assertTrue(result.finished)
        self.assertEqual(len(result.turns[0].actions), 1)
        self.assertFalse(os.path.exists(self.p("should_not_exist.txt")))

    def test_invalid_json_is_reported_and_loop_continues(self):
        actions = ["{ this is not valid json",
                   {"tool": "finish", "args": {"summary": "ok"}}]
        result = self._agent(actions, max_steps=5).run("t")
        self.assertFalse(result.turns[0].actions[0].ok)
        self.assertIn("invalid", result.turns[0].actions[0].observation)
        self.assertTrue(result.finished)

    def test_unknown_tool_is_reported(self):
        actions = [{"tool": "teleport", "args": {}},
                   {"tool": "finish", "args": {"summary": "ok"}}]
        result = self._agent(actions, max_steps=5).run("t")
        self.assertFalse(result.turns[0].actions[0].ok)
        self.assertIn("not a real tool", result.turns[0].actions[0].observation)

    def test_stops_at_step_budget_without_finish(self):
        actions = [{"tool": "list_directory", "args": {"path": self.dir}}]
        result = self._agent(actions, max_steps=3).run("loop forever")
        self.assertFalse(result.finished)
        self.assertEqual(len(result.turns), 3)

    def test_transcript_and_to_dict(self):
        actions = [{"tool": "finish", "args": {"summary": "done"}}]
        result = self._agent(actions).run("t")
        text = result.transcript()
        self.assertIn("TASK: t", text)
        self.assertIn("FINISHED: True", text)
        d = result.to_dict()
        self.assertEqual(d["task"], "t")
        self.assertIn("turns", d)
        self.assertIn("reasoning", d["turns"][0])  # reasoning trace present for logging


if __name__ == "__main__":
    unittest.main()
