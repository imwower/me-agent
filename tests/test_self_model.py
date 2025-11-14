import logging
import unittest
from datetime import datetime, timezone

from me_core.self_model.self_state import SelfState
from me_core.self_model.self_summarizer import summarize_self
from me_core.self_model.self_updater import aggregate_stats, update_from_event
from me_core.types import AgentEvent

# 为测试输出配置基础日志，便于观察状态更新情况
logging.basicConfig(level=logging.INFO)


class SelfStateTestCase(unittest.TestCase):
    """SelfState 的基础行为测试。"""

    def test_add_activity_with_clamp(self) -> None:
        """验证 add_activity 能正确裁剪历史长度。"""

        state = SelfState()
        for i in range(25):
            state.add_activity(f"activity-{i}", max_len=10)

        self.assertEqual(len(state.recent_activities), 10)
        # 应只保留最后 10 条记录
        self.assertEqual(state.recent_activities[0], "activity-15")
        self.assertEqual(state.recent_activities[-1], "activity-24")

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """验证 to_dict / from_dict 能够一致地序列化与反序列化。"""

        state = SelfState(
            identity="一个测试用的智能体",
            capabilities={"summarize": 0.8},
            focus_topics=["自我模型"],
            limitations=["不了解实时世界"],
            recent_activities=["完成一次总结"],
            needs=["需要更多数据"],
        )

        data = state.to_dict()
        restored = SelfState.from_dict(data)
        self.assertEqual(data, restored.to_dict())


class SelfUpdaterTestCase(unittest.TestCase):
    """自我状态更新逻辑测试。"""

    def _event(
        self,
        event_type: str,
        payload: dict,
    ) -> AgentEvent:
        """构造简单的 AgentEvent 供测试使用。"""

        return AgentEvent(timestamp=datetime.now(timezone.utc), event_type=event_type, payload=payload)

    def test_update_from_event_success_increase_capability(self) -> None:
        """任务成功事件应提升对应能力熟练度。"""

        state = SelfState()
        event = self._event(
            "task",
            {
                "kind": "task",
                "task_type": "summarize",
                "success": True,
                "topic": "自我模型",
            },
        )

        new_state = update_from_event(state, event)
        self.assertGreater(new_state.capabilities["summarize"], 0.5)
        self.assertIn("自我模型", new_state.focus_topics)
        self.assertTrue(any("成功完成任务" in x for x in new_state.recent_activities))

    def test_update_from_event_failure_decrease_capability_and_add_limitation(self) -> None:
        """任务失败事件应降低能力并记录局限。"""

        state = SelfState()
        event = self._event(
            "task",
            {
                "kind": "task",
                "task_type": "summarize",
                "success": False,
                "error": "在复杂长文档总结上表现不稳定",
            },
        )

        new_state = update_from_event(state, event)
        self.assertLess(new_state.capabilities["summarize"], 0.5)
        self.assertIn("在复杂长文档总结上表现不稳定", new_state.limitations)
        self.assertTrue(any("任务失败" in x for x in new_state.recent_activities))

    def test_aggregate_stats_updates_capabilities(self) -> None:
        """aggregate_stats 应根据历史成功率合理更新能力。"""

        state = SelfState()

        history = []
        # 构造 summarize 能力：3 次成功，1 次失败
        for success in [True, True, True, False]:
            history.append(
                self._event(
                    "task",
                    {
                        "kind": "task",
                        "task_type": "summarize",
                        "success": success,
                    },
                )
            )

        # 构造 generate_code 能力：1 次成功，4 次失败
        for success in [True, False, False, False, False]:
            history.append(
                self._event(
                    "task",
                    {
                        "kind": "task",
                        "task_type": "generate_code",
                        "success": success,
                    },
                )
            )

        new_state = aggregate_stats(state, history)

        summarize_level = new_state.capabilities["summarize"]
        generate_level = new_state.capabilities["generate_code"]

        # 成功率较高的 summarize 应明显优于 generate_code
        self.assertGreater(summarize_level, generate_level)
        # 失败偏多的能力应该被标记为局限
        self.assertTrue(
            any("generate_code" in x for x in new_state.limitations)
        )


class SelfSummarizerTestCase(unittest.TestCase):
    """自我总结逻辑测试。"""

    def test_summarize_self_contains_key_info(self) -> None:
        """summarize_self 输出应包含 identity、能力与需求信息。"""

        state = SelfState(
            identity="一个帮助用户思考和试验的 AI 助手",
            capabilities={
                "summarize": 0.9,
                "reflect": 0.7,
            },
            focus_topics=["自我模型", "驱动力"],
            limitations=["对最新现实世界信息不够了解"],
            recent_activities=["完成了一次自我总结"],
            needs=["需要更多真实使用场景的反馈"],
        )

        summary = summarize_self(state)

        self.assertIn("who_am_i", summary)
        self.assertIn("what_can_i_do", summary)
        self.assertIn("what_do_i_need", summary)

        # 三个字段都不应为空
        self.assertTrue(summary["who_am_i"].strip())
        self.assertTrue(summary["what_can_i_do"].strip())
        self.assertTrue(summary["what_do_i_need"].strip())

        # 文本中应包含 identity 片段、能力名和需求内容
        self.assertIn("AI 助手", summary["who_am_i"])
        self.assertIn("summarize", summary["what_can_i_do"])
        self.assertIn("需要更多真实使用场景的反馈", summary["what_do_i_need"])
        # 最近活动也应体现在“我能做什么”的描述中
        self.assertIn("完成了一次自我总结", summary["what_can_i_do"])


if __name__ == "__main__":
    unittest.main()
