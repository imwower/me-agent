"""批量运行 Scenario 并输出评估 + 内省结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from me_core.agent import SimpleAgent
from me_core.config import AgentConfig, load_agent_config
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.introspection import IntrospectionGenerator
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.tasks import ScenarioRegistry, run_scenario
from me_core.tools import EchoTool, FileReadTool, HttpGetTool, SelfDescribeTool, TimeTool
from me_core.world_model import SimpleWorldModel


def build_agent(cfg: AgentConfig) -> SimpleAgent:
    perception = TextPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem()
    tools = {
        "echo": EchoTool(),
        "time": TimeTool(),
        "http_get": HttpGetTool(),
        "file_read": FileReadTool(),
        "self_describe": SelfDescribeTool(self_model=self_model, world_model=world_model),
    }
    learner = SimpleLearner()
    dialogue_policy = RuleBasedDialoguePolicy()
    event_stream = EventStream()

    agent = SimpleAgent(
        perception=perception,
        world_model=world_model,
        self_model=self_model,
        drive_system=drive_system,
        tools=tools,
        learner=learner,
        dialogue_policy=dialogue_policy,
        event_stream=event_stream,
        config=cfg,
    )
    # 保证 introspection generator 可用
    agent.introspection_generator = IntrospectionGenerator(world_model, self_model, learner)  # type: ignore[assignment]
    return agent


def load_experiments(path: Path | None) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return [
            {"id": "baseline", "config": {}, "scenarios": ["self_intro", "ask_time"]},
            {"id": "no_curiosity", "config": {"enable_curiosity": False}, "scenarios": ["self_intro"]},
        ]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行多组 Scenario 评估实验")
    parser.add_argument("--config", type=str, default=None, help="实验配置 JSON 路径（可选）")
    parser.add_argument("--agent-config", type=str, default=None, help="Agent 配置 JSON 路径（可选）")
    parser.add_argument("--output", type=str, default="outputs/experiment_report.jsonl", help="输出 JSONL 路径")
    args = parser.parse_args()

    exp_defs = load_experiments(Path(args.config) if args.config else None)
    registry = ScenarioRegistry()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for exp in exp_defs:
            cfg = load_agent_config(args.agent_config)
            for k, v in exp.get("config", {}).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            agent = build_agent(cfg)
            scenario_ids = exp.get("scenarios") or registry.list_ids()
            for sid in scenario_ids:
                scenario = registry.get(sid)
                if scenario is None:
                    continue
                start_step = getattr(agent.world_model, "current_step", 0) + 1
                result = run_scenario(agent, scenario)
                end_step = getattr(agent.world_model, "current_step", start_step)
                introspection = None
                if getattr(cfg, "enable_introspection", True):
                    introspection = agent.introspect(scenario_id=scenario.id, start_step=start_step, end_step=end_step)
                record = {
                    "experiment_id": exp.get("id"),
                    "scenario_id": scenario.id,
                    "score": result.score,
                    "success": result.success,
                    "details": result.details,
                    "introspection": introspection.to_dict() if introspection else None,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"实验完成，结果已写入 {output_path}")  # noqa: T201


if __name__ == "__main__":
    main()
