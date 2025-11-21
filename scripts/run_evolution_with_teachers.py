"""简单种群演化 + Teacher 集成的示例脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.policy.agents import load_agent_spec_from_files, AgentSpec
from me_core.policy import AgentPolicy
from me_core.config import AgentConfig, load_agent_config
from me_core.population.population import AgentPopulation
from me_core.population.runner import evaluate_population
from me_core.teachers import DummyTeacher, TeacherManager
from me_core.teachers.factory import create_teacher_manager_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="运行带 Teacher 的种群演化示例")
    parser.add_argument("--scenarios", type=str, default="self_intro,ask_time", help="逗号分隔的 scenario id 列表")
    parser.add_argument("--config", type=str, default=None, help="Agent 配置 JSON 路径（可选）")
    parser.add_argument("--policy", type=str, default=None, help="策略配置 JSON 路径（可选）")
    parser.add_argument("--output", type=str, default="outputs/evolution_report.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--experiment-config", type=str, default=None, help="完整实验配置 JSON（可选）")
    args = parser.parse_args()

    teacher_cfg = {"use_real_teacher": False}
    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    population = AgentPopulation()

    if args.experiment_config:
        exp_data = json.loads(Path(args.experiment_config).read_text(encoding="utf-8"))
        backend_cfg = exp_data.get("backend", {})
        cfg_obj = load_agent_config(args.config)
        for k, v in backend_cfg.items():
            if hasattr(cfg_obj, k):
                setattr(cfg_obj, k, v)
        policy_path = args.policy
        scenario_ids = exp_data.get("population", {}).get("scenarios", scenario_ids)
        spec = AgentSpec(id="spec_base", config=cfg_obj, policy=AgentPolicy(), backend_info=None)
        population.register(spec)
        teacher_cfg = exp_data.get("teacher", teacher_cfg)
    else:
        spec = load_agent_spec_from_files("spec_base", args.config, args.policy)
        population.register(spec)

    tm = create_teacher_manager_from_config(teacher_cfg)

    evaluate_population(population, scenario_ids, teacher_manager=tm, output_path=Path(args.output))

    print(f"演化/评估完成，结果写入 {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
