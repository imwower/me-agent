"""高层调度脚本：发现仓库 / 跑 benchmark / DevLoop / Population。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent import SimpleAgent
from me_core.agent.multi_agent import MultiAgentCoordinator
from me_core.agent.roles import ROLES_DEFAULT
from me_core.config import load_agent_config
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel
from me_core.tasks import ScenarioRegistry, list_benchmark_scenarios, run_scenario
from me_core.workspace import Workspace, RepoSpec, scan_local_repo_for_tools


def _build_workspace(path: str | None) -> Workspace:
    if path and Path(path).exists():
        return Workspace.from_json(path)
    # fallback: 单仓库模式
    spec = RepoSpec(id="me-agent", name="me-agent", path=str(Path(__file__).resolve().parents[1]), allowed_paths=["."])
    return Workspace([spec])


def _build_agent() -> SimpleAgent:
    cfg = load_agent_config(None)
    perception = TextPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem()
    learner = SimpleLearner()
    dialogue_policy = RuleBasedDialoguePolicy()
    tools = {}
    return SimpleAgent(
        perception=perception,
        world_model=world_model,
        self_model=self_model,
        drive_system=drive_system,
        tools=tools,
        learner=learner,
        dialogue_policy=dialogue_policy,
    )


def run_benchmark(agent: SimpleAgent) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for sc in list_benchmark_scenarios():
        res = run_scenario(agent, sc)
        results[sc.id] = {"score": res.score, "success": res.success}
    return results


def run_devloop_multi(agent: SimpleAgent, scenarios: List[str]) -> List[Dict[str, Any]]:
    ma = MultiAgentCoordinator.from_single_agent(agent)
    registry = ScenarioRegistry()
    logs: List[Dict[str, Any]] = []
    for sid in scenarios:
        sc = registry.get(sid)
        if sc is None:
            continue
        logs.append(ma.run_devloop_task(sc))
    return logs


def auto_discover(root: str, output: str) -> None:
    root_path = Path(root).expanduser()
    profiles = []
    for p in root_path.glob("*"):
        if p.is_dir():
            profiles.append(scan_local_repo_for_tools(str(p)))
    data = [p.__dict__ for p in profiles]
    Path(output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已发现 {len(profiles)} 个仓库，写入 {output}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="总控 orchestrator")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "devloop", "population"])
    parser.add_argument("--use-brain", action="store_true")
    parser.add_argument("--use-multi-agent", action="store_true")
    parser.add_argument("--auto-discover-repos", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default="self_intro")
    args = parser.parse_args()

    if args.auto_discover_repos:
        auto_discover(args.auto_discover_repos, "configs/workspace.generated.json")

    workspace = _build_workspace(args.workspace)
    agent = _build_agent()

    report: Dict[str, Any] = {"mode": args.mode, "use_brain": args.use_brain, "use_multi_agent": args.use_multi_agent}

    if args.mode == "benchmark":
        report["benchmark"] = run_benchmark(agent)
    elif args.mode == "devloop":
        scenario_ids = [s for s in args.scenarios.split(",") if s]
        if args.use_multi_agent:
            report["devloop"] = run_devloop_multi(agent, scenario_ids)
        else:
            reg = ScenarioRegistry()
            logs = []
            for sid in scenario_ids:
                sc = reg.get(sid)
                if sc:
                    res = run_scenario(agent, sc)
                    logs.append({"scenario_id": sid, "score": res.score, "success": res.success})
            report["devloop"] = logs
    else:
        report["population"] = {"status": "not_implemented"}

    print(json.dumps(report, ensure_ascii=False, indent=2))  # noqa: T201


if __name__ == "__main__":
    from me_core.world_model import SimpleWorldModel  # noqa: F401  # needed for _build_agent

    main()
