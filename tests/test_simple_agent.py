import unittest

from me_core.agent import SimpleAgent
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.tools import EchoTool, TimeTool
from me_core.world_model import SimpleWorldModel


def build_test_agent() -> SimpleAgent:
    """构造一个用于测试的 SimpleAgent。"""

    perception = TextPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem()
    tools = {
        "echo": EchoTool(),
        "time": TimeTool(),
    }
    learner = SimpleLearner()
    dialogue_policy = RuleBasedDialoguePolicy()
    event_stream = EventStream()

    return SimpleAgent(
        perception=perception,
        world_model=world_model,
        self_model=self_model,
        drive_system=drive_system,
        tools=tools,
        learner=learner,
        dialogue_policy=dialogue_policy,
        event_stream=event_stream,
        agent_id="test_agent",
    )


class SimpleAgentTestCase(unittest.TestCase):
    """SimpleAgent 单步行为的基础测试。"""

    def test_step_generates_perception_and_reply(self) -> None:
        """一次文本输入应至少产生一个感知事件和一条回复。"""

        agent = build_test_agent()

        reply = agent.step("你好，测试 agent")
        self.assertIsInstance(reply, str)
        self.assertTrue(reply.strip())

        events = agent.event_stream.to_list()
        # 至少应包含一个感知事件
        self.assertGreaterEqual(len(events), 1)

    def test_time_keyword_triggers_time_tool(self) -> None:
        """包含时间关键词的输入应触发 TimeTool 调用。"""

        agent = build_test_agent()

        reply = agent.step("现在几点了？请告诉我当前的时间。")
        self.assertIsInstance(reply, str)
        self.assertTrue(reply.strip())

        # 驱动力系统应选择 call_tool 意图，并记录工具返回结果
        self.assertIsNotNone(agent.last_intent)
        self.assertEqual(agent.last_intent.kind, "call_tool")  # type: ignore[union-attr]
        self.assertEqual(agent.last_intent.target_tool, "time")  # type: ignore[union-attr]
        self.assertIsNotNone(agent.last_tool_result)

    def test_world_and_self_model_updated_after_multiple_steps(self) -> None:
        """多轮对话后，world_model 摘要与 self_model 状态应发生变化。"""

        agent = build_test_agent()

        # 进行多轮简单对话
        agent.step("第一轮对话。")
        agent.step("第二轮对话。")

        world_summary = agent.world_model.summarize()
        self.assertIn("events", world_summary)
        self.assertGreater(world_summary["events"]["total"], 0)  # type: ignore[index]

        # 自我模型应能给出一段非空的中文自述
        desc = agent.self_model.describe()
        self.assertTrue(desc.strip())


if __name__ == "__main__":
    unittest.main()

