from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import uuid

from me_core.agent.simple_agent import SimpleAgent
from me_core.agent.roles import AgentRoleConfig, ROLES_DEFAULT
from me_core.tasks import Scenario, run_scenario
from me_core.codetasks import CodeTask
from me_core.tasks.experiment_types import ExperimentScenario
from me_core.agent.messages import TaskMessage


@dataclass
class RoleAgent:
    role: AgentRoleConfig
    agent: SimpleAgent


class MultiAgentCoordinator:
    """
    角色化多 Agent 协作的简化协调器。
    """

    def __init__(self, roles: Dict[str, RoleAgent]) -> None:
        self.roles = roles

    @classmethod
    def from_single_agent(cls, agent: SimpleAgent) -> "MultiAgentCoordinator":
        role_agents = {rid: RoleAgent(role=cfg, agent=agent) for rid, cfg in ROLES_DEFAULT.items()}
        return cls(role_agents)

    def run_devloop_task(self, task: Scenario | CodeTask | ExperimentScenario) -> Dict[str, Any]:
        """
        按角色顺序执行任务：
        - planner: 确定是否需要 brain
        - brain: 可选地调用脑工具（当前仅记录意向）
        - coder: 执行 Scenario 或 CodeTask
        - tester: 运行测试（如有）
        - critic: 汇总结果
        """

        record: Dict[str, Any] = {"task_type": type(task).__name__, "roles": {}, "messages": []}
        messages: List[TaskMessage] = []

        # planner
        planner = self.roles.get("planner")
        need_brain = False
        if planner:
            text = "规划完成，建议调用脑态" if hasattr(task, "requires_brain_infer") else "常规执行"
            record["roles"]["planner"] = {"message": text}
            messages.append(
                TaskMessage(
                    id=str(uuid.uuid4()),
                    from_role="planner",
                    to_role="coder",
                    kind="plan",
                    content=text,
                )
            )
            need_brain = bool(getattr(task, "requires_brain_infer", False))

        # brain
        if need_brain and "brain" in self.roles:
            record["roles"]["brain"] = {"brain_action": "brain_infer_planned"}
            messages.append(
                TaskMessage(
                    id=str(uuid.uuid4()),
                    from_role="brain",
                    to_role="planner",
                    kind="brain_state",
                    content="脑态：准备探索模式",
                )
            )

        # coder
        coder = self.roles.get("coder")
        if coder:
            if isinstance(task, Scenario):
                res = run_scenario(coder.agent, task)
                record["roles"]["coder"] = {"score": res.score, "success": res.success}
                messages.append(
                    TaskMessage(
                        id=str(uuid.uuid4()),
                        from_role="coder",
                        to_role="tester",
                        kind="code_suggestion",
                        content=f"完成 scenario {task.id}",
                    )
                )
            else:
                record["roles"]["coder"] = {"status": "code task executed (stub)"}

        # tester
        tester = self.roles.get("tester")
        if tester:
            record["roles"]["tester"] = {"status": "tests planned"}
            messages.append(
                TaskMessage(
                    id=str(uuid.uuid4()),
                    from_role="tester",
                    to_role="critic",
                    kind="test_result",
                    content="tests planned",
                )
            )

        # critic
        critic = self.roles.get("critic")
        if critic:
            record["roles"]["critic"] = {"decision": "accept"}
            messages.append(
                TaskMessage(
                    id=str(uuid.uuid4()),
                    from_role="critic",
                    to_role="planner",
                    kind="critique",
                    content="接受本轮改动",
                )
            )

        record["messages"] = [m.__dict__ for m in messages]
        return record
