import unittest
from datetime import datetime, timezone

from me_core.types import (
    AgentEvent,
    AgentState,
    EventKind,
    EventSource,
    Genotype,
    Individual,
    ToolCall,
    ToolProgram,
    ToolResult,
    ToolStats,
)
from me_core.self_model.self_state import SelfState
from me_core.drives.drive_vector import DriveVector


class AgentEventTypesTestCase(unittest.TestCase):
    """AgentEvent / ToolCall / ToolResult 的基础类型与序列化测试。"""

    def test_agent_event_to_from_dict_roundtrip(self) -> None:
        """AgentEvent 的 to_dict / from_dict 应能互相还原关键信息。"""

        payload = {"kind": "perception", "text": "你好，世界"}
        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload=payload,
            source=EventSource.HUMAN.value,
            kind=EventKind.PERCEPTION,
            trace_id="trace-1",
        )

        data = event.to_dict()
        restored = AgentEvent.from_dict(data)

        self.assertEqual(restored.event_type, event.event_type)
        self.assertEqual(restored.payload, event.payload)
        self.assertEqual(restored.source, event.source)
        # kind 应保持一致
        self.assertEqual(
            restored.kind.value if restored.kind else None,
            event.kind.value if event.kind else None,
        )

        # pretty 与 __str__ 不应抛出异常，且返回非空字符串
        self.assertTrue(event.pretty())
        self.assertTrue(str(event))

    def test_agent_event_from_legacy_dict(self) -> None:
        """from_dict 应兼容仅包含 timestamp / event_type / payload 的旧格式。"""

        legacy = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "task",
            "payload": {"kind": "task", "task_type": "demo"},
        }
        event = AgentEvent.from_dict(legacy)

        self.assertEqual(event.event_type, "task")
        self.assertEqual(event.payload["task_type"], "demo")  # type: ignore[index]


class ToolTypesTestCase(unittest.TestCase):
    """ToolCall / ToolResult / ToolStats / ToolProgram 的基础行为测试。"""

    def test_tool_call_to_from_dict(self) -> None:
        """ToolCall 的 to_dict / from_dict 能互相还原。"""

        call = ToolCall(
            tool_name="echo",
            arguments={"text": "hello"},
            call_id="call-1",
        )

        data = call.to_dict()
        restored = ToolCall.from_dict(data)

        self.assertEqual(restored.tool_name, "echo")
        self.assertEqual(restored.arguments, {"text": "hello"})
        self.assertEqual(restored.call_id, "call-1")
        # 便捷别名字段也应工作正常
        self.assertEqual(restored.id, restored.call_id)
        self.assertEqual(restored.name, restored.tool_name)
        self.assertEqual(restored.args, restored.arguments)

        self.assertTrue(call.pretty())
        self.assertTrue(str(call))

    def test_tool_call_from_alt_field_names(self) -> None:
        """from_dict 应兼容 name / args / id / created_at 等备选字段名。"""

        data = {
            "name": "time",
            "args": {},
            "id": "xyz",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        call = ToolCall.from_dict(data)

        self.assertEqual(call.tool_name, "time")
        self.assertEqual(call.call_id, "xyz")

    def test_tool_result_to_from_dict(self) -> None:
        """ToolResult 的 to_dict / from_dict 能互相还原。"""

        result = ToolResult(
            call_id="call-1",
            success=True,
            output={"value": 42},
            error=None,
            meta={"debug": True},
        )

        data = result.to_dict()
        restored = ToolResult.from_dict(data)

        self.assertEqual(restored.call_id, "call-1")
        self.assertTrue(restored.success)
        self.assertEqual(restored.output, {"value": 42})
        self.assertEqual(restored.meta.get("debug"), True)

        self.assertTrue(result.pretty())
        self.assertTrue(str(result))

    def test_tool_stats_and_program_basic(self) -> None:
        """ToolStats 与 ToolProgram 应能正常实例化并更新。"""

        stats = ToolStats()
        self.assertEqual(stats.usage_count, 0)

        prog = ToolProgram(
            name="macro_move",
            dsl_source="MOVE_UP;MOVE_RIGHT",
            parents=["primitive_move"],
        )
        self.assertEqual(prog.name, "macro_move")
        self.assertEqual(prog.stats.usage_count, 0)

        # 模拟一次使用更新
        prog.stats.usage_count += 1
        prog.stats.success_count += 1
        prog.stats.avg_gain = 0.5
        prog.stats.last_used_step = 10

        self.assertEqual(prog.stats.usage_count, 1)
        self.assertEqual(prog.stats.success_count, 1)
        self.assertAlmostEqual(prog.stats.avg_gain, 0.5)
        self.assertEqual(prog.stats.last_used_step, 10)


class AgentStateAndPopulationTypesTestCase(unittest.TestCase):
    """AgentState / Genotype / Individual 的基础行为测试。"""

    def test_agent_state_aggregation(self) -> None:
        """AgentState 应能聚合 SelfState 与 DriveVector。"""

        self_state = SelfState(identity="一个用于测试的智能体")
        drives = DriveVector()

        state = AgentState(
            self_state=self_state,
            drives=drives,
            world_model_state={"param_count": 10},
            memory_state={"events": 5},
            tool_library_state={"tool_count": 2},
            global_step=42,
            env_state_summary={"level_id": "L1"},
        )

        self.assertEqual(state.self_state.identity, "一个用于测试的智能体")
        self.assertIn("param_count", state.world_model_state)
        self.assertIn("events", state.memory_state)
        self.assertIn("tool_count", state.tool_library_state)
        self.assertEqual(state.global_step, 42)
        self.assertIn("level_id", state.env_state_summary)

    def test_genotype_and_individual_basic(self) -> None:
        """Genotype / Individual 应能构造并挂接 AgentState。"""

        self_state = SelfState(identity="种群个体")
        drives = DriveVector(chat_level=0.3)
        agent_state = AgentState(self_state=self_state, drives=drives)

        genotype = Genotype(
            id="g-1",
            parent_ids=["g-0"],
            world_model_config={"layers": 2},
            learning_config={"lr": 0.01},
            drive_baseline={"chat_level": 0.3},
            tool_config={"max_tools": 8},
        )

        ind = Individual(
            id="ind-1",
            agent_state=agent_state,
            genotype=genotype,
            fitness=1.23,
            age=2,
            generation=1,
            parent_ids=["parent-1"],
            env_id="gridworld-v0",
            eval_count=3,
            frozen=True,
        )

        self.assertEqual(ind.id, "ind-1")
        self.assertEqual(ind.agent_state.self_state.identity, "种群个体")
        self.assertEqual(ind.genotype.id, "g-1")
        self.assertEqual(ind.genotype.world_model_config["layers"], 2)
        self.assertAlmostEqual(ind.fitness, 1.23)
        self.assertEqual(ind.age, 2)
        self.assertEqual(ind.generation, 1)
        self.assertEqual(ind.parent_ids, ["parent-1"])
        self.assertEqual(ind.env_id, "gridworld-v0")
        self.assertEqual(ind.eval_count, 3)
        self.assertTrue(ind.frozen)


if __name__ == "__main__":
    unittest.main()
