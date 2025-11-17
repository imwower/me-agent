from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core_env import BaseEnv, EnvObs, EnvStepReturn, PrimitiveFn

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


@dataclass
class GridWorldEnv(BaseEnv):
    """简单的二维网格环境。

    环境特性：
        - 状态为一个 w x h 的网格；
        - 智能体有一个当前位置 agent_pos；
        - 可以设定一个目标位置 goal_pos 以及若干墙壁 cells；
        - 每步动作包括：up/down/left/right；
        - 撞墙或越界时位置不变，并给出轻微负奖励；
        - 到达目标位置时给出正奖励并 episode 结束。

    观测格式（obs）：
        {
            "agent_pos": (x, y),
            "goal_pos": (x, y),
            "walls": [(x1, y1), ...],  # 只在 reset 时完整提供，后续可选
        }

    该环境主要用于：
        - Phase2 自监督 world_model 的测试；
        - 后续工具系统从行为轨迹中抽象出“路径工具”等模式。
    """

    width: int = 5
    height: int = 5
    start_pos: Position = (0, 0)
    goal_pos: Position = (4, 4)
    walls: List[Position] = field(default_factory=list)

    agent_pos: Position = field(init=False)

    def __post_init__(self) -> None:
        self.agent_pos = self.start_pos

    # BaseEnv 接口实现 -----------------------------------------------------

    def reset(self) -> EnvObs:
        """重置环境到起始位置，返回初始观测。"""

        self.agent_pos = self.start_pos
        logger.info("重置 GridWorldEnv: agent_pos=%s, goal_pos=%s", self.agent_pos, self.goal_pos)
        return self._build_obs(include_walls=True)

    def step(self, action: str) -> EnvStepReturn:
        """执行一个原语动作，并返回 (obs, reward, done, info)。"""

        old_pos = self.agent_pos
        new_pos = self._move(old_pos, action)

        if new_pos in self.walls or not self._in_bounds(new_pos):
            # 撞墙或越界：位置不变，给出轻微负奖励
            new_pos = old_pos
            reward = -0.1
            hit_wall = True
        else:
            reward = -0.01  # 每走一步略微惩罚，鼓励尽快到达目标
            hit_wall = False

        self.agent_pos = new_pos
        done = new_pos == self.goal_pos
        if done:
            reward = 1.0

        obs = self._build_obs(include_walls=False)
        info = {
            "hit_wall": hit_wall,
        }

        logger.info(
            "GridWorldEnv.step: action=%s, old_pos=%s, new_pos=%s, reward=%.3f, done=%s",
            action,
            old_pos,
            new_pos,
            reward,
            done,
        )
        return obs, reward, done, info

    def get_primitives(self) -> Dict[str, PrimitiveFn]:
        """返回一组原语动作包装器。

        这里只是简单将四个方向动作映射为不同字符串，并包装为无参数函数，
        便于工具系统在不直接依赖内部实现的情况下调用。
        """

        def make_primitive(action_name: str) -> PrimitiveFn:
            def _fn() -> EnvStepReturn:
                return self.step(action_name)

            return _fn

        return {
            "move_up": make_primitive("up"),
            "move_down": make_primitive("down"),
            "move_left": make_primitive("left"),
            "move_right": make_primitive("right"),
            "stay": make_primitive("stay"),
        }

    # 内部辅助方法 ---------------------------------------------------------

    def _in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _move(self, pos: Position, action: str) -> Position:
        x, y = pos
        if action == "up":
            return x, y - 1
        if action == "down":
            return x, y + 1
        if action == "left":
            return x - 1, y
        if action == "right":
            return x + 1, y
        # "stay" 或未知动作都视为不动
        return pos

    def _build_obs(self, include_walls: bool = False) -> Dict[str, object]:
        obs: Dict[str, object] = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
        }
        if include_walls:
            obs["walls"] = list(self.walls)
        return obs

