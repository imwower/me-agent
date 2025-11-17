import logging
import unittest

from me_core.agent import AgentCore
from me_core.self_model.self_state import SelfState
from me_core.drives.drive_vector import DriveVector

# 为测试输出配置基础日志，便于观察内部事件与状态变化
logging.basicConfig(level=logging.INFO)


class AgentCoreTestCase(unittest.TestCase):
    """AgentCore 在无环境场景下的最小行为测试。"""

    def test_step_generates_internal_event_and_updates_self_state(self) -> None:
        """每次 step 应生成一条内部事件并推动 SelfState 更新。"""

        core = AgentCore()

        # 初始状态记录
        initial_identity = core.self_state.identity

        event = core.step()
        self.assertEqual(event.event_type, "internal")
        self.assertEqual(event.payload.get("kind"), "internal")  # type: ignore[union-attr]

        # 事件应被写入事件流
        events = core.recent_events()
        self.assertGreaterEqual(len(events), 1)

        # 自我状态应仍然存在且可通过 summarize_self 正常描述
        summary = core.summarize_self()
        self.assertIn("who_am_i", summary)
        self.assertTrue(summary["who_am_i"].strip())
        # identity 在仅有内部事件时不应被清空
        self.assertIn(initial_identity, summary["who_am_i"])

    def test_as_agent_state_and_custom_init(self) -> None:
        """AgentCore 应能用自定义 SelfState / DriveVector 初始化，并正确导出 AgentState。"""

        self_state = SelfState(identity="一个测试用的 AgentCore")
        drives = DriveVector(chat_level=0.7)

        core = AgentCore(self_state=self_state, drives=drives)

        agent_state = core.as_agent_state()
        self.assertEqual(agent_state.self_state.identity, "一个测试用的 AgentCore")
        self.assertEqual(agent_state.drives.chat_level, 0.7)
        self.assertIn("event_count", agent_state.memory_state)


if __name__ == "__main__":
    unittest.main()

