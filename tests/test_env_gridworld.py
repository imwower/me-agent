import logging
import unittest

from envs.gridworld import GridWorldEnv

# 为测试输出配置基础日志，便于观察 GridWorldEnv 行为
logging.basicConfig(level=logging.INFO)


class GridWorldEnvTestCase(unittest.TestCase):
    """GridWorldEnv 的基础行为测试。"""

    def test_reset_and_step(self) -> None:
        """reset 应设置初始位置，step 应能在网格内移动。"""

        env = GridWorldEnv(width=3, height=3, start_pos=(0, 0), goal_pos=(2, 2))

        obs = env.reset()
        self.assertEqual(obs["agent_pos"], (0, 0))  # type: ignore[index]
        self.assertEqual(obs["goal_pos"], (2, 2))  # type: ignore[index]

        # 尝试向右移动几步，不应越界
        obs1, reward1, done1, info1 = env.step("right")
        self.assertEqual(obs1["agent_pos"], (1, 0))  # type: ignore[index]
        self.assertFalse(done1)
        self.assertIsInstance(reward1, float)
        self.assertIn("hit_wall", info1)

        obs2, _, _, _ = env.step("down")
        self.assertEqual(obs2["agent_pos"], (1, 1))  # type: ignore[index]

    def test_get_primitives(self) -> None:
        """get_primitives 应返回可执行的原语动作集合。"""

        env = GridWorldEnv()
        primitives = env.get_primitives()

        self.assertIn("move_up", primitives)
        self.assertIn("move_down", primitives)

        # 调用一个原语动作，应与直接调用 step 行为兼容
        env.reset()
        _, _, _, _ = primitives["move_right"]()
        # agent_pos 应该发生变化
        obs_after, _, _, _ = env.step("stay")
        self.assertNotEqual(obs_after["agent_pos"], (0, 0))  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()

