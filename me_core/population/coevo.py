from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from me_core.memory.log_index import LogIndex
from me_core.population.population import AgentPopulation
from me_core.tasks.generated.pool import TaskPool
from me_core.tasks.train_schedule import TrainSchedule
from me_core.policy.agents import AgentSpec


@dataclass
class CoEvoState:
    generation: int
    agent_specs: List[AgentSpec]
    snn_train_schedules: List[TrainSchedule] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


class CoEvoPlanner:
    def __init__(self, population: AgentPopulation, task_pool: TaskPool, log_index: LogIndex) -> None:
        self.population = population
        self.task_pool = task_pool
        self.log_index = log_index

    def propose_next_round(
        self,
        prev_state: Optional[CoEvoState],
        prev_results: Dict[str, Any],
        max_tasks: int = 5,
        max_train_steps: int | None = None,
    ) -> CoEvoState:
        gen = 0 if prev_state is None else prev_state.generation + 1
        specs = self.population.get_specs()
        # 简化：按日志中最近实验分数决定任务难度
        recent = self.log_index.query(kinds=["experiment"], max_results=5)
        difficulty = 1
        for r in recent:
            if float(r.get("score", 1.0)) < 0.5:
                difficulty = 1
            else:
                difficulty = 2
        tasks = self.task_pool.sample_tasks(difficulty_range=(difficulty, difficulty + 2), max_count=max_tasks)
        schedule = TrainSchedule(
            id=f"sched_{gen}",
            repo_id="self-snn",
            tasks=tasks,
            config_path="configs/agency.yaml",
            output_dir=f"runs/coevo/gen{gen}",
            max_epochs=1,
            max_steps=max_train_steps,
        )
        history = []
        if prev_state:
            history = list(prev_state.history)
        history.append({"generation": gen, "results": prev_results})
        return CoEvoState(generation=gen, agent_specs=specs, snn_train_schedules=[schedule], history=history)
