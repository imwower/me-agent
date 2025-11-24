"""自学/自改流水线：运行场景 -> 内省/Teacher -> 生成 CodeTask -> 调用 Code-LLM -> 写回并跑测试。"""

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
from me_core.config import AgentConfig, load_agent_config
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.introspection import IntrospectionGenerator, IntrospectionLog
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.policy import load_policy_from_file, policy_to_dict
from me_core.policy.agents import AgentSpec
from me_core.self_model import SimpleSelfModel
from me_core.tasks import ScenarioRegistry, run_scenario
from me_core.teachers.factory import create_teacher_manager_from_config
from me_core.teachers.types import TeacherInput
from me_core.tools import (
    EchoTool,
    FileReadTool,
    HttpGetTool,
    ReadFileTool,
    RunTestsTool,
    SelfDescribeTool,
    TimeTool,
    WriteFileTool,
)
from me_core.codetasks import CodeTaskPlanner, PromptGenerator
from me_core.workspace import RepoSpec, Workspace
from me_core.world_model import SimpleWorldModel
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


def _apply_llm_output(workspace: Workspace, repo_id: str, output: str) -> List[str]:
    """解析 Code-LLM 输出并写回文件，返回被修改的路径列表。"""

    try:
        data = json.loads(output)
    except Exception:
        return []

    if isinstance(data, dict) and "file_changes" in data:
        items = data.get("file_changes") or []
    elif isinstance(data, dict) and "path" in data and "content" in data:
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    write_tool = WriteFileTool(workspace)
    changed: List[str] = []
    for item in items:
        try:
            path = item.get("path")
            content = item.get("content", "")
            if not path:
                continue
            write_tool.run({"repo_id": repo_id, "path": path, "content": content})
            changed.append(path)
        except Exception:
            continue
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
) -> Dict[str, Any]:
    registry = ScenarioRegistry()
    teacher_manager = create_teacher_manager_from_config(teacher_cfg)
    agent = build_agent(agent_spec)
    codellm = CodeLLMClient(codellm_cfg)
    planner = CodeTaskPlanner()
    prompt_generator = PromptGenerator()
    test_tool = RunTestsTool(workspace)

    results: List[Dict[str, Any]] = []
    output.parent.mkdir(parents=True, exist_ok=True)

    for sid in scenario_ids:
        scenario = registry.get(sid)
        if scenario is None:
            continue
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
        )
        teacher_outputs = teacher_manager.gather_advice(teacher_input)

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
            prompt = prompt_generator.generate(task, contents)
            llm_output = codellm.complete(prompt)
            changed_files = _apply_llm_output(workspace, repo_id, llm_output)
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
    return {"results": results, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行一次 DevLoop，自我改写代码 + 测试")
    parser.add_argument("--workspace", type=str, default=None, help="workspace 配置 JSON 路径")
    parser.add_argument("--repo-id", type=str, default=None, help="目标仓库 id，默认取 workspace 中第一个")
    parser.add_argument("--agent-config", type=str, default=None, help="Agent 配置 JSON 路径")
    parser.add_argument("--policy", type=str, default=None, help="Policy 配置 JSON 路径")
    parser.add_argument("--teacher-config", type=str, default=None, help="Teacher 配置 JSON 路径")
    parser.add_argument("--codellm-config", type=str, default=None, help="Code-LLM 配置 JSON 路径")
    parser.add_argument("--scenarios", type=str, default="self_intro", help="要运行的 scenario id，逗号分隔")
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

    summary = run_devloop(
        workspace=workspace,
        repo_id=repo_id,
        scenario_ids=scenario_ids,
        agent_spec=agent_spec,
        teacher_cfg=teacher_cfg,
        codellm_cfg=codellm_cfg,
        output=output_path,
    )
    print(f"DevLoop 完成，结果写入 {summary['output']}")  # noqa: T201


if __name__ == "__main__":
    main()
