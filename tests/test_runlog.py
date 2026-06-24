import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from vibeharness.agent import Action, RunResult, Turn
from vibeharness.config import Config
from vibeharness.runlog import RunLogger

STAMP = datetime(2026, 1, 2, 3, 4, 5)


class RunLoggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, finished=True) -> RunResult:
        turn = Turn(index=1, reasoning="<think>checking</think>",
                    raw_action='[{"tool":"validate","args":{"summary":"done"}}]')
        turn.actions.append(Action("validate", {"summary": "done"},
                                   "validation PASSED — looks good", ok=True, final=True))
        return RunResult(task="demo", turns=[turn], finished=finished, final_summary="looks good",
                         validations=[{"turn": 1, "passed": True, "reason": "looks good",
                                       "reasoning": "<think>ok</think>"}])

    def test_writes_json_and_md_into_hidden_dir(self):
        logger = RunLogger(self.workspace, STAMP)
        path = logger.write("demo", Config(), self._result())
        self.assertTrue(path.exists())
        self.assertEqual(path.parent.name, ".vibe")
        self.assertTrue(path.with_suffix(".md").exists())

    def test_json_contains_reasoning_and_validations(self):
        path = RunLogger(self.workspace, STAMP).write("demo", Config(), self._result())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["task"], "demo")
        self.assertTrue(data["finished"])
        self.assertEqual(data["turns"][0]["reasoning"], "<think>checking</think>")
        self.assertEqual(data["validations"][0]["passed"], True)

    def test_streaming_overwrites_same_file(self):
        # writing twice (e.g. per turn) reuses the same timestamped file
        logger = RunLogger(self.workspace, STAMP)
        p1 = logger.write("demo", Config(), self._result(finished=False))
        p2 = logger.write("demo", Config(), self._result(finished=True))
        self.assertEqual(p1, p2)
        data = json.loads(p2.read_text(encoding="utf-8"))
        self.assertTrue(data["finished"])   # reflects the latest state


if __name__ == "__main__":
    unittest.main()
