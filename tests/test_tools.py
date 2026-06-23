import os
import tempfile
import unittest

from vibeharness.filesystem import FileSystem
from vibeharness.fs_tools import (FinishTool, ListDirectoryTool, ManagePathTool,
                                  ReadFileTool, SearchTool, WriteFileTool,
                                  build_default_tools)


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fs = FileSystem()

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, *parts):
        return os.path.join(self.dir, *parts)

    def test_write_then_read_observations(self):
        w = WriteFileTool(self.fs).run({"path": self.p("a.txt"), "content": "hi there"})
        self.assertTrue(w.ok)
        self.assertIn("you wrote the file", w.observation)

        r = ReadFileTool(self.fs, 1000).run({"path": self.p("a.txt")})
        self.assertTrue(r.ok)
        self.assertIn("hi there", r.observation)

    def test_write_append_observation_verb(self):
        WriteFileTool(self.fs).run({"path": self.p("a.txt"), "content": "a"})
        res = WriteFileTool(self.fs).run({"path": self.p("a.txt"), "content": "b", "mode": "append"})
        self.assertIn("appended to", res.observation)

    def test_read_missing_is_error(self):
        res = ReadFileTool(self.fs, 1000).run({"path": self.p("nope.txt")})
        self.assertFalse(res.ok)
        self.assertIn("error", res.observation)

    def test_list_directory(self):
        WriteFileTool(self.fs).run({"path": self.p("a.txt"), "content": "x"})
        res = ListDirectoryTool(self.fs, 1000).run({"path": self.dir})
        self.assertTrue(res.ok)
        self.assertIn("a.txt", res.observation)

    def test_search_tool(self):
        WriteFileTool(self.fs).run({"path": self.p("a.txt"), "content": "find ME"})
        res = SearchTool(self.fs, 1000).run({"query": "me", "path": self.dir})
        self.assertTrue(res.ok)
        self.assertIn("a.txt", res.observation)

    def test_manage_make_delete_move(self):
        mp = ManagePathTool(self.fs)
        self.assertTrue(mp.run({"action": "make_directory", "path": self.p("d")}).ok)
        WriteFileTool(self.fs).run({"path": self.p("d", "a.txt"), "content": "x"})
        moved = mp.run({"action": "move", "path": self.p("d", "a.txt"),
                        "destination": self.p("b.txt")})
        self.assertTrue(moved.ok)
        self.assertIn("moved", moved.observation)
        self.assertTrue(mp.run({"action": "delete", "path": self.p("b.txt")}).ok)

    def test_manage_move_without_destination_fails(self):
        res = ManagePathTool(self.fs).run({"action": "move", "path": self.p("a.txt")})
        self.assertFalse(res.ok)

    def test_finish_is_final(self):
        res = FinishTool().run({"summary": "all done"})
        self.assertTrue(res.ok)
        self.assertTrue(res.is_final)
        self.assertIn("all done", res.observation)

    def test_default_toolset_names(self):
        names = {t.name for t in build_default_tools(self.fs, 1000)}
        self.assertEqual(names, {"list_directory", "read_file", "write_file",
                                 "search", "manage_path", "finish"})


if __name__ == "__main__":
    unittest.main()
