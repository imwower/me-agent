from __future__ import annotations

import unittest

from me_core.codetasks import CodeTask, PromptGenerator


class PromptGeneratorTestCase(unittest.TestCase):
    def test_prompt_contains_key_sections(self) -> None:
        task = CodeTask(
            id="ct-1",
            repo_id="repo",
            title="demo",
            description="修复 world_model 的统计问题",
            files_to_read=["a.py"],
            constraints=["仅使用标准库"],
            acceptance_criteria=["单测通过"],
        )
        contents = {"a.py": "print('hello')"}
        prompt = PromptGenerator().generate(task, contents)
        self.assertIn("ct-1", prompt)
        self.assertIn("a.py", prompt)
        self.assertIn("单测通过", prompt)
        self.assertIn("print('hello')", prompt)


if __name__ == "__main__":
    unittest.main()
