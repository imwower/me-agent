import logging
import unittest

from me_core.tools.executor_stub import ToolExecutorStub
from me_core.tools.registry import ToolInfo

# 为测试输出配置基础日志
logging.basicConfig(level=logging.INFO)


class CodexToolStubTestCase(unittest.TestCase):
    """Codex 工具桩实现的行为测试。"""

    def test_codex_tool_returns_deterministic_answer(self) -> None:
        """同一主题和提示多次调用，应得到一致的结果。"""

        tool = ToolInfo(
            name="codex",
            type="llm",
            cost=0.4,
            description="Codex 风格回答桩。",
            good_for=[],
        )
        executor = ToolExecutorStub()

        args = {"topic": "自我模型", "prompt": "请解释什么是自我模型"}
        r1 = executor.execute(tool, args)
        r2 = executor.execute(tool, args)

        self.assertTrue(r1.success)
        self.assertEqual(r1.summary, r2.summary)
        self.assertEqual(r1.details, r2.details)
        self.assertIn("Codex 桩实现", r1.details["simulated_answer"])
        self.assertIn("自我模型", r1.details["simulated_answer"])


if __name__ == "__main__":
    unittest.main()

