import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from vibeharness.agent import Action, RunResult, Turn
from vibeharness.config import Config
from vibeharness.runlog import RunLogger


class RunLoggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _sample_result(self) -> RunResult:
        turn = Turn(index=1, reasoning="<think>I will write the file</think>",
                    raw_action='[{"tool":"finish","args":{"summary":"done"}}]')
        turn.actions.append(Action("finish", {"summary": "done"}, "you finished the task: done",
                                   ok=True, final=True))
        return RunResult(task="demo", turns=[turn], finished=True, final_summary="done")

    def test_writes_json_and_md_into_hidden_dir(self):
        path = RunLogger(self.workspace).write(
            "demo", Config(), self._sample_result(), datetime(2026, 1, 2, 3, 4, 5))
        self.assertTrue(path.exists())
        self.assertEqual(path.parent.name, ".vibe")
        self.assertTrue(path.with_suffix(".md").exists())

    def test_json_contains_reasoning_trace(self):
        path = RunLogger(self.workspace).write(
            "demo", Config(), self._sample_result(), datetime(2026, 1, 2, 3, 4, 5))
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["task"], "demo")
        self.assertTrue(data["finished"])
        self.assertIn("model", data)
        self.assertEqual(data["turns"][0]["reasoning"], "<think>I will write the file</think>")
        self.assertEqual(data["turns"][0]["actions"][0]["tool"], "finish")


if __name__ == "__main__":
    unittest.main()
