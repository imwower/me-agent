from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

from me_core.workspace import FileEdit, RepoSpec, Workspace


class WorkspaceRepoTestCase(unittest.TestCase):
    def test_read_write_and_apply_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            spec = RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])
            ws = Workspace([spec])
            repo = ws.get_repo("repo")

            repo.write_file("foo.txt", "hello")
            self.assertIn("hello", repo.read_file("foo.txt"))

            repo.apply_edits([FileEdit(path="foo.txt", old_snippet="hello", new_snippet="world", reason="test")])
            self.assertIn("world", repo.read_file("foo.txt"))

            rc, out, err = repo.run_command([sys.executable, "-c", "print('ok')"])
            self.assertEqual(0, rc)
            self.assertIn("ok", out)

    def test_disallow_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            spec = RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["subdir"])
            ws = Workspace([spec])
            repo = ws.get_repo("repo")
            with self.assertRaises(PermissionError):
                repo.write_file("outside.txt", "boom")


if __name__ == "__main__":
    unittest.main()
