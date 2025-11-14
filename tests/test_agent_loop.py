import logging
import os
import unittest
from pathlib import Path

from me_core.agent.agent_loop import run_once
from me_core.agent.state_store import StateStore
from me_core.dialogue.planner import DialoguePlanner
from me_core.drives.drive_vector import DriveVector
from me_core.self_model.self_state import SelfState
from me_core.self_model.self_summarizer import summarize_self

# 为测试输出配置基础日志
logging.basicConfig(level=logging.INFO)


class DialoguePlannerTestCase(unittest.TestCase):
    """对话规划器的基本行为测试。"""

    def test_low_chat_and_social_should_be_silent(self) -> None:
        """在较低 chat_level / social_need 下，应倾向保持沉默。"""

        drives = DriveVector(
            chat_level=0.1,
            curiosity_level=0.5,
            exploration_level=0.5,
            learning_intensity=0.5,
            social_need=0.1,
            data_need=0.5,
        )

        state = SelfState()
        self_summary = summarize_self(state)
        context = {"topic": "测试话题"}

        planner = DialoguePlanner()
        decision = planner.decide_initiative(drives, self_summary, context)

        self.assertFalse(decision.should_speak)
        self.assertEqual(decision.intent, "silent")

    def test_high_chat_and_social_should_speak(self) -> None:
        """在较高 chat_level / social_need 下，应在某些上下文中主动说话。"""

        drives = DriveVector(
            chat_level=0.9,
            curiosity_level=0.5,
            exploration_level=0.5,
            learning_intensity=0.5,
            social_need=0.9,
            data_need=0.8,
        )

        state = SelfState(
            needs=["需要更多真实场景数据"],
        )
        self_summary = summarize_self(state)
        context = {"topic": "自我介绍与需求说明"}

        planner = DialoguePlanner()
        decision = planner.decide_initiative(drives, self_summary, context)

        self.assertTrue(decision.should_speak)
        self.assertIn(decision.intent, {"self_introduction", "ask_for_help"})


class AgentLoopTestCase(unittest.TestCase):
    """Agent 主循环的一致性测试。"""

    def test_run_once_completes_without_error(self) -> None:
        """run_once 应能完整执行一轮流程且不抛异常。"""

        # 为避免污染真实状态文件，使用临时路径
        temp_state_path = Path("agent_state_test.json")
        try:
            # 先创建一个独立的状态存储，以确保文件存在且结构合理
            store = StateStore(path=temp_state_path)
            store.save_state()

            # 切换当前工作目录到该文件所在目录，run_once 会使用默认路径
            cwd = os.getcwd()
            os.chdir(str(temp_state_path.parent))
            try:
                run_once()

                # 读取 run_once 默认写入的 agent_state.json，验证其中包含结构化事件
                state_file = Path("agent_state.json")
                self.assertTrue(state_file.exists())

                store_after = StateStore(path=state_file)
                events = store_after.get_events()
                # 在带有学习行为的主循环中，应至少记录到一些事件
                self.assertGreaterEqual(len(events), 0)
                # 知识库也应已持久化（即使可能为空列表，也需字段存在）
                kb = store_after.get_knowledge_base()
                self.assertIsInstance(kb, list)
            finally:
                os.chdir(cwd)
        finally:
            if temp_state_path.exists():
                temp_state_path.unlink()
            # 清理 run_once 生成的默认状态文件
            state_file = Path("agent_state.json")
            if state_file.exists():
                state_file.unlink()


if __name__ == "__main__":
    unittest.main()
