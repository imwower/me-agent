from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

from me_core.tools import ApplyPatchTool, ReadFileTool, RunCommandTool, RunTestsTool, WriteFileTool
from me_core.workspace import RepoSpec, Workspace


class CodeAndRunToolsTestCase(unittest.TestCase):
    def test_read_write_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            spec = RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])
            ws = Workspace([spec])
            write_tool = WriteFileTool(ws)
            read_tool = ReadFileTool(ws)
            patch_tool = ApplyPatchTool(ws)

            write_tool.run({"repo_id": spec.id, "path": "a.txt", "content": "hello"})
            res = read_tool.run({"repo_id": spec.id, "path": "a.txt"})
            self.assertIn("hello", res["content"])

            patch_tool.run(
                {
                    "repo_id": spec.id,
                    "path": "a.txt",
                    "old": "hello",
                    "new": "world",
                    "reason": "test",
                }
            )
            res2 = read_tool.run({"repo_id": spec.id, "path": "a.txt"})
            self.assertIn("world", res2["content"])

    def test_run_command_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            spec = RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])
            ws = Workspace([spec])
            # 写入一个最小可运行的单测
            test_file = repo_path / "test_sample.py"
            test_file.write_text(
                "import unittest\n\nclass Demo(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n",
                encoding="utf-8",
            )

            cmd_tool = RunCommandTool(ws)
            res = cmd_tool.run({"repo_id": spec.id, "cmd": [sys.executable, "-c", "print('cmd')"]})
            self.assertEqual(0, res["returncode"])
            self.assertIn("cmd", res["stdout"])

            tests_tool = RunTestsTool(ws)
            res2 = tests_tool.run(
                {"repo_id": spec.id, "command": [sys.executable, "-m", "unittest", "discover"]}
            )
            self.assertTrue(res2["success"])
            self.assertIn("OK", res2["stdout"])


if __name__ == "__main__":
    unittest.main()
