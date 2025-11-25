"""联合进化主流程（简化版）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.population.population import AgentPopulation
from me_core.policy.agents import load_agent_spec_from_files
from me_core.population.coevo import CoEvoPlanner, CoEvoState
from me_core.memory.log_index import LogIndex
from me_core.tasks.generated.pool import TaskPool


def main() -> None:
    parser = argparse.ArgumentParser(description="运行联合进化流程（demo）")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--tasks-root", type=str, default="data/generated_tasks")
    parser.add_argument(
        "--gens",
        type=int,
        default=2,
        help="联合进化的代数（别名 --generations）",
    )
    parser.add_argument(
        "--generations",
        type=int,
        dest="gens",
        help="同 --gens，兼容旧 CLI 调用",
    )
    parser.add_argument("--output", type=str, default="logs/coevo.jsonl")
    args = parser.parse_args()

    pop = AgentPopulation()
    pop.register(load_agent_spec_from_files("agent1", None, None))
    task_pool = TaskPool(args.tasks_root)
    log_index = LogIndex("logs")
    planner = CoEvoPlanner(pop, task_pool, log_index)

    state: CoEvoState | None = None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for gen in range(args.gens):
        results = {"agent_scores": {"agent1": 0.5 + 0.1 * gen}}
        state = planner.propose_next_round(state, results)
        out_path.write_text("", encoding="utf-8") if not out_path.exists() else None
        with out_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "generation": state.generation,
                        "schedules": [s.id for s in state.snn_train_schedules],
                        "results": results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"CoEvo 运行完成，输出 {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
