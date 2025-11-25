"""R15 小型闭环：学习->评估->自改->训练调度。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.agent import SimpleAgent
from me_core.config import AgentConfig, load_agent_config
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.policy.schema import AgentPolicy
from me_core.tasks.generated import TaskGenerator, TaskTemplate
from me_core.tasks.train_schedule import TrainSchedule, dump_train_schedule
from me_core.tools import EchoTool, TimeTool
from me_core.world_model import SimpleWorldModel
from me_ext.dialogue import RealDialogueLLM


def _build_agent(cfg: AgentConfig) -> SimpleAgent:
    perception = TextPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem(policy_config=AgentPolicy())
    tools = {"echo": EchoTool(), "time": TimeTool()}
    learner = SimpleLearner()
    dialogue_llm = RealDialogueLLM(cfg.dialogue_llm) if getattr(cfg, "use_llm_dialogue", False) else None
    dialogue_policy = RuleBasedDialoguePolicy(policy_config=None, agent_config=cfg, dialogue_llm=dialogue_llm)
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
        agent_id="small-loop",
    )
    return agent


def _default_templates() -> List[TaskTemplate]:
    return [
        TaskTemplate(
            id="tpl-mm",
            kind="multimodal",
            description="轻量多模态理解任务",
            input_schema={},
            output_schema={},
            difficulty=1,
        ),
        TaskTemplate(
            id="tpl-code",
            kind="codefix",
            description="小型代码修复任务",
            input_schema={},
            output_schema={},
            difficulty=1,
        ),
    ]


def _run_simple_benchmark(agent: SimpleAgent) -> Dict[str, Any]:
    reply = agent.step("你好，做个自检如何？")
    return {"score": 1.0 if reply else 0.2, "reply": reply or ""}


def _call_self_snn(schedule_path: Path, train_script: Path) -> Dict[str, Any]:
    cmd = [sys.executable, str(train_script), "--train-schedule", str(schedule_path), "--dry-run"]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = (completed.stdout or "").strip()
        return {"status": "ok", "raw": output}
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def run_small_full_loop(
    workspace: str | None,
    agent_config_path: str | None,
    snn_config: str,
    snn_output: Path,
    snn_train_script: Path,
    use_brain: bool,
    use_llm_dialogue: bool,
) -> Dict[str, Any]:
    cfg = load_agent_config(agent_config_path)
    cfg.use_llm_dialogue = cfg.use_llm_dialogue or use_llm_dialogue
    agent = _build_agent(cfg)

    benchmark_before = _run_simple_benchmark(agent)

    templates = _default_templates()
    generator = TaskGenerator(templates)
    intro_logs: List[Dict[str, Any]] = []
    introspection = agent.introspect("small-loop", start_step=0, end_step=agent.world_model.current_step)
    if introspection:
        intro_logs.append(introspection.to_dict())
    tasks = generator.generate_tasks_from_gaps(intro_logs, [benchmark_before], brain_graph=None, max_new_tasks=3)

    snn_config_path = Path(snn_config)
    if not snn_config_path.is_absolute():
        candidate = snn_train_script.parent.parent / snn_config_path
        if candidate.exists():
            snn_config_path = candidate

    schedule = TrainSchedule(
        id="small-loop",
        repo_id="self-snn",
        tasks=tasks,
        config_path=str(snn_config_path),
        output_dir=str(snn_output),
        max_epochs=1,
    )
    snn_output.mkdir(parents=True, exist_ok=True)
    schedule_path = snn_output / "train_schedule.json"
    dump_train_schedule(schedule, str(schedule_path))

    snn_result = _call_self_snn(schedule_path, snn_train_script)

    if tasks:
        agent.learner.policy_learner.record_outcome("curiosity.min_concept_count", reward=0.2, success=True)
    policy_updates: Dict[str, Any] = {}
    policy = getattr(agent.drive_system, "policy_config", None)
    if policy is not None:
        policy_updates = agent.learner.policy_learner.propose_updates(policy)
        agent.learner.policy_learner.apply_updates(policy, policy_updates)

    benchmark_after = _run_simple_benchmark(agent)

    summary = {
        "workspace": workspace,
        "benchmark_before": benchmark_before,
        "benchmark_after": benchmark_after,
        "train_schedule": str(schedule_path),
        "snn_result": snn_result,
        "policy_updates": policy_updates,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default=None, help="workspace 配置路径，可选")
    parser.add_argument("--agent-config", type=str, default=None, help="AgentConfig 路径，可选")
    parser.add_argument("--snn-config", type=str, default="configs/s0_minimal.yaml", help="self-snn 训练基础 config")
    parser.add_argument("--snn-output", type=str, default="outputs/snn_small_loop", help="self-snn 输出目录")
    parser.add_argument(
        "--snn-train-script",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "self-snn" / "scripts" / "train_from_schedule.py"),
    )
    parser.add_argument("--use-brain", action="store_true")
    parser.add_argument("--use-llm-dialogue", action="store_true")
    args = parser.parse_args()

    run_small_full_loop(
        workspace=args.workspace,
        agent_config_path=args.agent_config,
        snn_config=args.snn_config,
        snn_output=Path(args.snn_output),
        snn_train_script=Path(args.snn_train_script),
        use_brain=args.use_brain,
        use_llm_dialogue=args.use_llm_dialogue,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
