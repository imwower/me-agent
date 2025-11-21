from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from me_core.agent import SimpleAgent
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.introspection import IntrospectionGenerator
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.policy.agents import AgentSpec
from me_core.policy.applier import apply_policy_patches
from me_core.tasks import ScenarioRegistry, run_scenario
from me_core.tools import EchoTool, FileReadTool, HttpGetTool, SelfDescribeTool, TimeTool
from me_core.world_model import SimpleWorldModel
from me_core.teachers.manager import TeacherManager
from me_core.teachers.types import TeacherInput

from .population import AgentPopulation
from .types import AgentFitness


def build_agent_from_spec(spec: AgentSpec) -> SimpleAgent:
    perception = TextPerception()
    world_model = SimpleWorldModel()
    drive_system = SimpleDriveSystem()
    # 将 policy 配置注入驱动力/对话策略
    drive_system.policy_config = getattr(spec, "policy", None)
    dialogue_policy = RuleBasedDialoguePolicy(policy_config=getattr(spec, "policy", None))
    self_model = None
    try:
        from me_core.self_model import SimpleSelfModel
        self_model = SimpleSelfModel()
    except Exception:
        self_model = None
    learner = SimpleLearner()
    tools = {
        "echo": EchoTool(),
        "time": TimeTool(),
        "http_get": HttpGetTool(),
        "file_read": FileReadTool(),
        "self_describe": SelfDescribeTool(self_model=self_model, world_model=world_model),
    }
    event_stream = EventStream()

    agent = SimpleAgent(
        perception=perception,
        world_model=world_model,
        self_model=self_model,  # type: ignore[arg-type]
        drive_system=drive_system,
        tools=tools,
        learner=learner,
        dialogue_policy=dialogue_policy,
        event_stream=event_stream,
        config=spec.config,
    )
    agent.introspection_generator = IntrospectionGenerator(world_model, self_model, learner)  # type: ignore[assignment]
    return agent


def evaluate_population(
    population: AgentPopulation,
    scenario_ids: List[str],
    teacher_manager: Optional[TeacherManager],
    generations: int = 1,
    output_path: Optional[Path] = None,
) -> Dict[str, AgentFitness]:
    registry = ScenarioRegistry()
    results: Dict[str, AgentFitness] = {}
    out_file = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_path.open("w", encoding="utf-8")

    for spec in population.get_specs():
        agent = build_agent_from_spec(spec)
        scenario_scores: Dict[str, float] = {}
        introspection_notes: List[str] = []

        for sid in scenario_ids:
            sc = registry.get(sid)
            if sc is None:
                continue
            start_step = getattr(agent.world_model, "current_step", 0) + 1
            res = run_scenario(agent, sc)
            scenario_scores[sid] = res.score
            end_step = getattr(agent.world_model, "current_step", start_step)
            if getattr(spec.config, "enable_introspection", True) and agent.introspection_generator:
                log = agent.introspect(scenario_id=sc.id, start_step=start_step, end_step=end_step)
                if log:
                    introspection_notes.append(log.summary)
                    if teacher_manager is not None:
                        ti = TeacherInput(
                            scenario_id=sc.id,
                            episodes=[],
                            introspection=log,
                            current_config={"policy": spec.policy},
                        )
                        outs = teacher_manager.gather_advice(ti)
                        patches = teacher_manager.aggregate_patches(outs)
                        spec.policy = apply_policy_patches(spec.policy, patches)

        overall = sum(scenario_scores.values()) / len(scenario_scores) if scenario_scores else 0.0
        results[spec.id] = AgentFitness(
            spec_id=spec.id,
            scenario_scores=scenario_scores,
            overall_score=overall,
            introspection_summaries=introspection_notes,
        )
        if out_file:
            out_file.write(
                json.dumps(
                    {
                        "spec_id": spec.id,
                        "scores": scenario_scores,
                        "overall_score": overall,
                        "introspection": introspection_notes,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if out_file:
        out_file.close()
    return results


__all__ = ["evaluate_population", "build_agent_from_spec"]
