import unittest

from vibeharness.web import BrowseTool


class FakeCli:
    """Stand-in for PlaywrightCli: records calls, returns a scripted result."""
    def __init__(self, ok=True, output="### Page\nok"):
        self.ok, self.output = ok, output
        self.calls = []

    def run(self, *args):
        self.calls.append(list(args))
        return self.ok, self.output


class BrowseToolTest(unittest.TestCase):
    def _tool(self, ok=True, output="### Page URL: https://example.com"):
        self.cli = FakeCli(ok=ok, output=output)
        return BrowseTool(self.cli, observation_limit=1000)

    def test_goto_maps_to_cli_args(self):
        res = self._tool().run({"action": "goto", "url": "https://example.com"})
        self.assertTrue(res.ok)
        self.assertEqual(self.cli.calls, [["goto", "https://example.com"]])
        self.assertIn("navigated to", res.observation)
        self.assertIn("https://example.com", res.observation)

    def test_snapshot_needs_no_params(self):
        res = self._tool(output="- heading 'Example'").run({"action": "snapshot"})
        self.assertTrue(res.ok)
        self.assertEqual(self.cli.calls, [["snapshot"]])

    def test_fill_requires_target_and_text(self):
        res = self._tool().run({"action": "fill", "target": "e3"})
        self.assertFalse(res.ok)
        self.assertIn("text", res.observation)
        self.assertEqual(self.cli.calls, [])  # never reached the CLI

    def test_click_maps_target(self):
        self._tool().run({"action": "click", "target": "e6"})
        self.assertEqual(self.cli.calls, [["click", "e6"]])

    def test_eval_maps_expression(self):
        self._tool().run({"action": "eval", "expression": "() => document.title"})
        self.assertEqual(self.cli.calls, [["eval", "() => document.title"]])

    def test_unknown_action_is_error(self):
        res = self._tool().run({"action": "teleport"})
        self.assertFalse(res.ok)
        self.assertIn("unknown browser action", res.observation)

    def test_cli_failure_is_reported(self):
        res = self._tool(ok=False, output="net::ERR_NAME_NOT_RESOLVED").run(
            {"action": "goto", "url": "https://nope.invalid"})
        self.assertFalse(res.ok)
        self.assertIn("failed", res.observation)

    def test_output_is_truncated(self):
        tool = BrowseTool(FakeCli(output="x" * 5000), observation_limit=100)
        res = tool.run({"action": "snapshot"})
        self.assertIn("truncated", res.observation)

    def test_browse_schema_branch_present(self):
        tool = self._tool()
        schema = tool.call_schema()
        self.assertEqual(schema["properties"]["tool"]["const"], "browse")
        self.assertIn("action", schema["properties"]["args"]["properties"])


if __name__ == "__main__":
    unittest.main()
