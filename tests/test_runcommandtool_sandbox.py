from __future__ import annotations

import unittest

from me_core.tools.runtools import RunCommandTool
from me_core.workspace import RepoSpec, Workspace


class RunCommandSandboxTestCase(unittest.TestCase):
    def test_blocked_command(self) -> None:
        spec = RepoSpec(id="r", name="r", path=".", allowed_paths=["."])
        tool = RunCommandTool(Workspace([spec]), blocked_commands=["rm"])
        res = tool.run({"repo_id": "r", "cmd": ["rm", "-rf", "/"]})
        self.assertEqual(res["returncode"], -1)


if __name__ == "__main__":
    unittest.main()
