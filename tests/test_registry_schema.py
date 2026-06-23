import unittest

from vibeharness.config import Config
from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import build_default_tools
from vibeharness.registry import ToolRegistry
from vibeharness.tools import Param, Tool, ToolResult


class _NoParamTool(Tool):
    name = "noop"
    description = "does nothing"

    @property
    def parameters(self):
        return []

    def run(self, args):
        return ToolResult(True, "you did nothing")


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(build_default_tools(FileSystem(), 1000))

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            ToolRegistry([_NoParamTool(), _NoParamTool()])

    def test_empty_registry_rejected(self):
        with self.assertRaises(ValueError):
            ToolRegistry([])

    def test_get_and_names(self):
        self.assertIsNotNone(self.registry.get("write_file"))
        self.assertIsNone(self.registry.get("does_not_exist"))
        self.assertIn("finish", self.registry.names())

    def test_action_schema_is_oneof_over_all_tools(self):
        schema = self.registry.action_schema()
        self.assertIn("oneOf", schema)
        self.assertEqual(len(schema["oneOf"]), len(self.registry.all()))
        consts = {b["properties"]["tool"]["const"] for b in schema["oneOf"]}
        self.assertEqual(consts, set(self.registry.names()))

    def test_each_branch_requires_tool_and_args(self):
        for branch in self.registry.action_schema()["oneOf"]:
            self.assertEqual(branch["required"], ["tool", "args"])
            self.assertIn("args", branch["properties"])

    def test_finish_branch_requires_summary(self):
        branch = next(b for b in self.registry.action_schema()["oneOf"]
                      if b["properties"]["tool"]["const"] == "finish")
        self.assertEqual(branch["properties"]["args"]["required"], ["summary"])

    def test_docs_mention_every_tool(self):
        docs = self.registry.docs()
        for name in self.registry.names():
            self.assertIn(name, docs)


class ParamTest(unittest.TestCase):
    def test_required_param_doc_and_schema(self):
        p = Param("path", "string", "a path")
        self.assertIn("required", p.doc())
        self.assertEqual(p.schema()["type"], "string")

    def test_enum_param(self):
        p = Param("mode", "string", "how", required=False, default="overwrite",
                  enum=("overwrite", "append"))
        self.assertIn("optional", p.doc())
        self.assertIn("overwrite", p.doc())
        self.assertEqual(p.schema()["enum"], ["overwrite", "append"])


if __name__ == "__main__":
    unittest.main()
