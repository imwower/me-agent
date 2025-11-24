"""自学/自改流水线：运行场景 -> 内省/Teacher -> 生成 CodeTask -> 调用 Code-LLM -> 写回并跑测试。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent import SimpleAgent
from me_core.config import AgentConfig, load_agent_config
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.introspection import IntrospectionGenerator, IntrospectionLog
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.policy import load_policy_from_file, policy_to_dict
from me_core.policy.agents import AgentSpec
from me_core.policy.applier import apply_policy_patches
from me_core.self_model import SimpleSelfModel
from me_core.tasks import (
    Scenario,
    ScenarioRegistry,
    run_scenario,
    ExperimentScenario,
    ExperimentScenarioRegistry,
    ExperimentStep,
    run_experiment_scenario,
    evaluate_experiment_results,
)
from me_core.teachers.factory import create_teacher_manager_from_config
from me_core.teachers.types import TeacherInput
from me_core.tools import (
    EchoTool,
    FileReadTool,
    HttpGetTool,
    DumpBrainGraphTool,
    EvalBrainEnergyTool,
    EvalBrainMemoryTool,
    BrainInferTool,
    ReadFileTool,
    RunTestsTool,
    SelfDescribeTool,
    TimeTool,
    WriteFileTool,
)
from me_core.codetasks import CodeTaskPlanner, PromptGenerator
from me_core.codetasks import apply_config_patches
from me_core.workspace import RepoSpec, Workspace
from me_core.world_model import SimpleWorldModel
from me_core.brain import BrainSnapshot
from me_ext.codellm import CodeLLMClient


def _load_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_agent(spec: AgentSpec) -> SimpleAgent:
    cfg = spec.config
    perception = TextPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem(policy_config=spec.policy)
    tools = {
        "echo": EchoTool(),
        "time": TimeTool(),
        "http_get": HttpGetTool(),
        "file_read": FileReadTool(),
        "self_describe": SelfDescribeTool(self_model=self_model, world_model=world_model),
    }
    learner = SimpleLearner()
    dialogue_policy = RuleBasedDialoguePolicy(policy_config=spec.policy)
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
        agent_id=spec.id,
    )
    agent.introspection_generator = IntrospectionGenerator(world_model, self_model, learner)  # type: ignore[assignment]
    return agent


def _default_workspace() -> Workspace:
    repo_root = ROOT_DIR
    spec = RepoSpec(id="me-agent", name="me-agent", path=str(repo_root), allowed_paths=["."])
    return Workspace([spec])


def _build_workspace(path: str | None) -> Workspace:
    if path:
        return Workspace.from_json(path)
    return _default_workspace()


def _make_agent_spec(agent_config_path: str | None, policy_path: str | None) -> AgentSpec:
    cfg = load_agent_config(agent_config_path)
    policy = load_policy_from_file(policy_path)
    return AgentSpec(id="devloop-agent", config=cfg, policy=policy)


def _apply_llm_output(
    workspace: Workspace,
    repo_id: str,
    output: str,
    output_format: str,
    write_tool: WriteFileTool,
) -> List[str]:
    """解析 Code-LLM 输出并写回文件，返回被修改的路径列表。"""

    changed: List[str] = []
    try:
        if output_format == "files":
            data = json.loads(output)
            if isinstance(data, dict) and "file_changes" in data:
                items = data.get("file_changes") or []
            elif isinstance(data, dict) and "path" in data and "content" in data:
                items = [data]
            elif isinstance(data, list):
                items = data
            else:
                raise ValueError("unsupported files format")
            for item in items:
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                content = item.get("content", "")
                if not path:
                    continue
                res = write_tool.run({"repo_id": repo_id, "path": path, "content": content})
                if res.get("ok"):
                    changed.append(path)
            return changed
        if output_format == "json_diff":
            data = json.loads(output)
            changes = data.get("changes") if isinstance(data, dict) else data
            if not isinstance(changes, list):
                raise ValueError("json_diff changes not list")
            for item in changes:
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                new_content = item.get("new_content")
                if not path or new_content is None:
                    continue
                res = write_tool.run({"repo_id": repo_id, "path": path, "content": str(new_content)})
                if res.get("ok"):
                    changed.append(path)
            return changed
        if output_format == "raw_diff":
            raise ValueError("raw_diff not supported in apply step")
    except Exception:
        return []
    return changed


def _collect_file_contents(workspace: Workspace, repo_id: str, paths: List[str]) -> Dict[str, str]:
    reader = ReadFileTool(workspace)
    contents: Dict[str, str] = {}
    for p in paths:
        try:
            res = reader.run({"repo_id": repo_id, "path": p})
            contents[p] = res.get("content", "")
        except Exception:
            contents[p] = ""
    return contents


def run_devloop(
    workspace: Workspace,
    repo_id: str,
    scenario_ids: List[str],
    agent_spec: AgentSpec,
    teacher_cfg: Dict[str, Any],
    codellm_cfg: Dict[str, Any],
    output: Path,
    experiment_scenarios: Optional[List[ExperimentScenario]] = None,
    brain_mode: bool = False,
) -> Dict[str, Any]:
    registry = ScenarioRegistry()
    exp_registry = ExperimentScenarioRegistry()
    if experiment_scenarios:
        for sc in experiment_scenarios:
            exp_registry.register(sc)
    teacher_manager = create_teacher_manager_from_config(teacher_cfg)
    agent = build_agent(agent_spec)
    codellm = CodeLLMClient(codellm_cfg)
    planner = CodeTaskPlanner()
    prompt_generator = PromptGenerator()
    test_tool = RunTestsTool(workspace)
    write_tool = WriteFileTool(
        workspace,
        max_lines=agent_spec.config.max_write_lines_per_file if agent_spec.config else 500,
        max_files_per_run=agent_spec.config.max_files_per_run if agent_spec.config else 10,
    )
    brain_summary: str | None = None
    brain_snapshots: List[BrainSnapshot] = []

    def _run_online_brain_infer(scenario: Scenario) -> None:
        """在脑模式下执行一次在线脑推理，并写入 world/self。"""

        nonlocal brain_summary
        brain_repos = workspace.get_brain_repos()
        target_repo = brain_repos[0] if brain_repos else None
        if not target_repo:
            return
        infer_tool = BrainInferTool(workspace)
        text_summary = scenario.steps[0].user_input if getattr(scenario, "steps", None) else scenario.description
        cfg_path = None
        if getattr(target_repo, "meta", None):
            cfg_path = target_repo.meta.get("brain_config") or target_repo.meta.get("default_config")
        res = infer_tool.run(
            {
                "repo_id": target_repo.id,
                "task_id": scenario.id,
                "text": text_summary or scenario.description,
                "features": {"uncertainty": 1.0} if getattr(scenario, "requires_brain_infer", False) else {},
                "config_path": cfg_path or "configs/agency.yaml",
            }
        )
        snap_data = res.get("snapshot")
        if not snap_data:
            return
        snapshot = BrainSnapshot(
            repo_id=snap_data.get("repo_id", target_repo.id),
            region_activity=snap_data.get("region_activity", {}) or {},
            global_metrics=snap_data.get("global_metrics", {}) or {},
            memory_summary=snap_data.get("memory_summary", {}) or {},
            decision_hint=snap_data.get("decision_hint", {}) or {},
            created_at=float(snap_data.get("created_at", time.time()) or time.time()),
        )
        if hasattr(agent.world_model, "update_brain_snapshot"):
            agent.world_model.update_brain_snapshot(snapshot)  # type: ignore[arg-type]
        if hasattr(agent.self_model, "observe_brain_snapshot"):
            agent.self_model.observe_brain_snapshot(snapshot)  # type: ignore[arg-type]
        brain_snapshots.append(snapshot)
        nonlocal brain_summary
        if not brain_summary:
            mode = snapshot.decision_hint.get("mode") if isinstance(snapshot.decision_hint, dict) else None
            brain_summary = f"在线脑态: mode={mode}, κ={snapshot.global_metrics.get('branching_kappa')}"

    # brain mode: 尝试获取脑结构/能耗/记忆信息
    if brain_mode:
        brain_repos = workspace.get_brain_repos()
        target_repo = brain_repos[0] if brain_repos else None
        if target_repo:
            graph_tool = DumpBrainGraphTool(workspace)
            energy_tool = EvalBrainEnergyTool(workspace)
            memory_tool = EvalBrainMemoryTool(workspace)
            g_res = graph_tool.run({"repo_id": target_repo.id})
            brain_summary = g_res.get("summary")
            energy_res = energy_tool.run({"repo_id": target_repo.id})
            memory_res = memory_tool.run({"repo_id": target_repo.id})
            # 记录到 output
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"brain": g_res, "energy": energy_res, "memory": memory_res}, ensure_ascii=False) + "\n")

    results: List[Dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)

    for sid in scenario_ids:
        scenario = registry.get(sid)
        if scenario is None:
            continue
        if brain_mode or getattr(scenario, "requires_brain_infer", False):
            _run_online_brain_infer(scenario)
        start_step = getattr(agent.world_model, "current_step", 0) + 1
        task_result = run_scenario(agent, scenario)
        end_step = getattr(agent.world_model, "current_step", start_step)
        introspection: IntrospectionLog | None = None
        if getattr(agent_spec.config, "enable_introspection", True):
            introspection = agent.introspect(
                scenario_id=scenario.id,
                start_step=start_step,
                end_step=end_step,
                notes="devloop",
            )

        episodes = []
        if getattr(agent, "episodic_memory", None):
            try:
                episodes = agent.episodic_memory.recent_episodes(5)  # type: ignore[assignment]
            except Exception:
                episodes = []

        teacher_input = TeacherInput(
            scenario_id=scenario.id,
            episodes=episodes,
            introspection=introspection,
            current_config={
                "agent_config": agent_spec.config.__dict__,
                "policy": policy_to_dict(agent_spec.policy),
            },
            notes="devloop",
            brain_graph=None,
            brain_snapshot=brain_snapshots[-1] if brain_snapshots else None,
        )
        teacher_outputs = teacher_manager.gather_advice(teacher_input)
        # 应用策略补丁（仅内存）
        patches = teacher_manager.aggregate_patches(teacher_outputs)
        agent_spec.policy = apply_policy_patches(agent_spec.policy, patches)

        tasks = planner.plan_tasks(
            repo_id=repo_id,
            introspection=introspection,
            teacher_outputs=teacher_outputs,
            task_result=task_result,
        )
        task_logs: List[Dict[str, Any]] = []
        for task in tasks:
            paths_to_read = list(set(task.files_to_read + task.files_to_edit))
            contents = _collect_file_contents(workspace, repo_id, paths_to_read)
            prompt = prompt_generator.generate(task, contents, brain_summary=brain_summary)
            llm_output = codellm.complete(prompt)
            changed_files = _apply_llm_output(
                workspace,
                repo_id,
                llm_output,
                codellm_cfg.get("output_format", getattr(codellm, "output_format", "files")),
                write_tool,
            )
            test_res = test_tool.run({"repo_id": repo_id, "command": task.test_command})
            task_logs.append(
                {
                    "task_id": task.id,
                    "changed_files": changed_files,
                    "llm_output": llm_output[:2000],
                    "test_success": test_res.get("success"),
                    "test_summary": test_res.get("summary"),
                }
            )

        record = {
            "scenario_id": scenario.id,
            "score": task_result.score,
            "success": task_result.success,
            "introspection": introspection.to_dict() if introspection else None,
            "teacher_outputs": [o.advice_text for o in teacher_outputs],
            "tasks": task_logs,
        }
        results.append(record)
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 可选实验流程
    experiment_results_all: List[Dict[str, Any]] = []
    if experiment_scenarios:
        for exp_sc in exp_registry.list_ids():
            sc = exp_registry.get(exp_sc)
            if sc is None:
                continue
            exp_results = run_experiment_scenario(workspace, sc)
            score = evaluate_experiment_results(exp_results, sc.eval_formula or "0.0")
            exp_record = {
                "experiment_id": sc.id,
                "score": score,
                "steps": [
                    {"repo_id": r.step.repo_id, "kind": r.step.kind, "metrics": r.metrics, "returncode": r.returncode}
                    for r in exp_results
                ],
            }
            experiment_results_all.append(exp_record)
            if getattr(agent_spec.config, "enable_introspection", True):
                agent.introspect(
                    scenario_id=sc.id,
                    start_step=getattr(agent.world_model, "current_step", 0),
                    end_step=getattr(agent.world_model, "current_step", 0),
                    notes="experiment",
                    experiment_results=exp_results,
                )
            # Teacher 建议 + ConfigPatch
            ti = TeacherInput(
                scenario_id=sc.id,
                episodes=[],
                introspection=None,
                current_config={
                    "agent_config": agent_spec.config.__dict__,
                    "policy": policy_to_dict(agent_spec.policy),
                    "config_path": sc.steps[0].command[0] if sc.steps else "",
                },
                experiment_results=exp_results,
                notes="experiment",
            )
            outs = teacher_manager.gather_advice(ti)
            patches = teacher_manager.aggregate_patches(outs)
            agent_spec.policy = apply_policy_patches(agent_spec.policy, patches)
            config_patches = []
            for o in outs:
                config_patches.extend(o.config_patches)
            if config_patches:
                apply_config_patches(workspace, config_patches)
            with output.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"experiment": exp_record, "teacher": [o.advice_text for o in outs]}, ensure_ascii=False) + "\n")

    brain_snapshots_dict = [
        {
            "repo_id": s.repo_id,
            "region_activity": s.region_activity,
            "global_metrics": s.global_metrics,
            "decision_hint": s.decision_hint,
            "created_at": s.created_at,
        }
        for s in brain_snapshots
    ]
    return {"results": results, "experiments": experiment_results_all, "output": str(output), "brain_snapshots": brain_snapshots_dict}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行一次 DevLoop，自我改写代码 + 测试")
    parser.add_argument("--workspace", type=str, default=None, help="workspace 配置 JSON 路径")
    parser.add_argument("--repo-id", type=str, default=None, help="目标仓库 id，默认取 workspace 中第一个")
    parser.add_argument("--agent-config", type=str, default=None, help="Agent 配置 JSON 路径")
    parser.add_argument("--policy", type=str, default=None, help="Policy 配置 JSON 路径")
    parser.add_argument("--teacher-config", type=str, default=None, help="Teacher 配置 JSON 路径")
    parser.add_argument("--codellm-config", type=str, default=None, help="Code-LLM 配置 JSON 路径")
    parser.add_argument("--scenarios", type=str, default="self_intro", help="要运行的 scenario id，逗号分隔")
    parser.add_argument("--experiment-scenarios", type=str, default="", help="实验场景 id（逗号），为空则不跑实验")
    parser.add_argument("--brain-mode", action="store_true", help="启用脑结构自改模式")
    parser.add_argument("--output", type=str, default="outputs/devloop_report.jsonl", help="输出 JSONL 路径")
    args = parser.parse_args()

    workspace = _build_workspace(args.workspace)
    repo_list = workspace.list_repos()
    if not repo_list:
        raise SystemExit("workspace 中未定义任何仓库")
    repo_id = args.repo_id or repo_list[0].id
    agent_spec = _make_agent_spec(args.agent_config, args.policy)
    teacher_cfg = _load_json(args.teacher_config)
    codellm_cfg = _load_json(args.codellm_config)
    scenario_ids = [s for s in args.scenarios.split(",") if s]
    output_path = Path(args.output)
    exp_ids = [s for s in args.experiment_scenarios.split(",") if s]

    experiment_scenarios: List[ExperimentScenario] = []
    if exp_ids:
        # 尝试根据 workspace meta 构造简单实验场景
        targets = workspace.get_experiment_targets() or repo_list
        target_repo = targets[0] if targets else repo_list[0]
        for sid in exp_ids:
            step = ExperimentStep(
                repo_id=target_repo.id,
                kind="train",
                command=target_repo.meta.get("default_train_cmd", ["python", "-c", 'print("{\\"loss\\\":0.1}")']),
                parse_mode="json",
                metrics_keys=["loss"],
            )
            experiment_scenarios.append(
                ExperimentScenario(
                    id=sid,
                    name=sid,
                    description="auto experiment",
                    steps=[step],
                    eval_formula="1 - train_loss" if "train_loss" in step.metrics_keys else "0.0",
                )
            )

    summary = run_devloop(
        workspace=workspace,
        repo_id=repo_id,
        scenario_ids=scenario_ids,
        agent_spec=agent_spec,
        teacher_cfg=teacher_cfg,
        codellm_cfg=codellm_cfg,
        output=output_path,
        experiment_scenarios=experiment_scenarios,
        brain_mode=args.brain_mode,
    )
    print(f"DevLoop 完成，结果写入 {summary['output']}")  # noqa: T201


if __name__ == "__main__":
    main()
