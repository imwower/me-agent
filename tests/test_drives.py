import logging
import unittest

from me_core.drives.drive_update import apply_user_command, implicit_adjust
from me_core.drives.drive_vector import DriveVector


# 为测试输出配置基础日志，便于调试驱动力变化过程
logging.basicConfig(level=logging.INFO)


class DriveVectorTestCase(unittest.TestCase):
    """DriveVector 的基础行为测试。"""

    def test_drive_vector_clamp_limits(self) -> None:
        """验证 clamp 能够将越界数值裁剪到 [0,1]。"""

        drives = DriveVector(
            chat_level=-0.5,
            curiosity_level=1.5,
            exploration_level=0.0,
            learning_intensity=2.0,
            social_need=-1.0,
            data_need=1.2,
        )

        drives.clamp()

        self.assertEqual(drives.chat_level, 0.0)
        self.assertEqual(drives.curiosity_level, 1.0)
        self.assertEqual(drives.exploration_level, 0.0)
        self.assertEqual(drives.learning_intensity, 1.0)
        self.assertEqual(drives.social_need, 0.0)
        self.assertEqual(drives.data_need, 1.0)

    def test_drive_vector_as_dict_and_from_dict(self) -> None:
        """验证 as_dict / from_dict 能够一致地序列化与反序列化。"""

        drives = DriveVector(
            chat_level=0.3,
            curiosity_level=0.7,
            exploration_level=0.4,
            learning_intensity=0.9,
            social_need=0.1,
            data_need=0.8,
        )

        data = drives.as_dict()
        restored = DriveVector.from_dict(data)

        self.assertEqual(data, restored.as_dict())


class DriveUpdateTestCase(unittest.TestCase):
    """驱动力更新逻辑的行为测试。"""

    def test_apply_user_command_chatty(self) -> None:
        """“多陪我聊天”类指令应提高聊天与社交相关驱动力。"""

        original = DriveVector(
            chat_level=0.1,
            curiosity_level=0.2,
            exploration_level=0.3,
            learning_intensity=0.4,
            social_need=0.1,
            data_need=0.5,
        )

        updated = apply_user_command(original, "多陪我聊天")

        self.assertGreater(updated.chat_level, original.chat_level)
        self.assertGreater(updated.social_need, original.social_need)
        # 其他无关字段应保持不变
        self.assertEqual(updated.curiosity_level, original.curiosity_level)
        self.assertEqual(
            updated.exploration_level, original.exploration_level
        )
        self.assertEqual(
            updated.learning_intensity, original.learning_intensity
        )
        self.assertEqual(updated.data_need, original.data_need)

    def test_apply_user_command_quiet(self) -> None:
        """“今天先安静点”应降低聊天与社交相关驱动力。"""

        original = DriveVector(
            chat_level=0.8,
            curiosity_level=0.5,
            exploration_level=0.5,
            learning_intensity=0.5,
            social_need=0.8,
            data_need=0.5,
        )

        updated = apply_user_command(original, "今天先安静点")

        self.assertLess(updated.chat_level, original.chat_level)
        self.assertLess(updated.social_need, original.social_need)

    def test_apply_user_command_exploration(self) -> None:
        """“多探索新东西”应提高好奇心与探索欲。"""

        original = DriveVector(
            chat_level=0.5,
            curiosity_level=0.2,
            exploration_level=0.2,
            learning_intensity=0.5,
            social_need=0.5,
            data_need=0.5,
        )

        updated = apply_user_command(original, "今天多探索点新东西")

        self.assertGreater(updated.curiosity_level, original.curiosity_level)
        self.assertGreater(
            updated.exploration_level, original.exploration_level
        )

    def test_apply_user_command_stable(self) -> None:
        """“先稳一点别乱折腾”应降低探索欲与学习强度。"""

        original = DriveVector(
            chat_level=0.5,
            curiosity_level=0.5,
            exploration_level=0.8,
            learning_intensity=0.9,
            social_need=0.5,
            data_need=0.5,
        )

        updated = apply_user_command(original, "先稳一点别乱折腾")

        self.assertLess(
            updated.exploration_level, original.exploration_level
        )
        self.assertLess(
            updated.learning_intensity, original.learning_intensity
        )

    def test_implicit_adjust_smooth_increase_chat(self) -> None:
        """高 user_response_ratio 应缓慢提升聊天相关驱动力。"""

        drives = DriveVector(
            chat_level=0.1,
            curiosity_level=0.5,
            exploration_level=0.5,
            learning_intensity=0.5,
            social_need=0.1,
            data_need=0.5,
        )

        feedback = {"user_response_ratio": 1.0}

        # 第一次调整应略微提升
        first = implicit_adjust(drives, feedback)
        self.assertGreater(first.chat_level, drives.chat_level)
        self.assertGreater(first.social_need, drives.social_need)
        # 且变化幅度不应过大（验证“平滑”）
        self.assertLess(first.chat_level - drives.chat_level, 0.5)

        # 多次迭代后，驱动力应进一步提升
        current = first
        for _ in range(10):
            current = implicit_adjust(current, feedback)

        self.assertGreater(current.chat_level, first.chat_level)
        self.assertGreater(current.social_need, first.social_need)

    def test_implicit_adjust_reduce_exploration_on_failure(self) -> None:
        """learning_success 为 0 且探索值较高时，应缓慢降低探索欲。"""

        drives = DriveVector(
            chat_level=0.5,
            curiosity_level=0.5,
            exploration_level=0.9,
            learning_intensity=0.5,
            social_need=0.5,
            data_need=0.5,
        )

        feedback = {"learning_success": 0.0}

        first = implicit_adjust(drives, feedback)
        # 第一次调用应略微降低探索值
        self.assertLess(first.exploration_level, drives.exploration_level)
        self.assertGreater(first.exploration_level, 0.7)

        # 多次迭代后，探索欲应继续降低，但不会瞬间跌到 0
        current = first
        for _ in range(10):
            current = implicit_adjust(current, feedback)

        self.assertLess(current.exploration_level, first.exploration_level)
        self.assertGreater(current.exploration_level, 0.3)

    def test_implicit_adjust_reduce_chat_on_low_response(self) -> None:
        """当用户几乎不回应时，应逐步降低话痨度与社交需求。"""

        drives = DriveVector(
            chat_level=0.8,
            curiosity_level=0.5,
            exploration_level=0.5,
            learning_intensity=0.5,
            social_need=0.8,
            data_need=0.5,
        )

        feedback = {"user_response_ratio": 0.0}

        first = implicit_adjust(drives, feedback)
        # 第一次调用应略微降低聊天与社交相关驱动
        self.assertLess(first.chat_level, drives.chat_level)
        self.assertLess(first.social_need, drives.social_need)

        # 多次迭代后，话痨度与社交需求应继续降低，但不会瞬间跌到 0
        current = first
        for _ in range(10):
            current = implicit_adjust(current, feedback)

        self.assertLess(current.chat_level, first.chat_level)
        self.assertLess(current.social_need, first.social_need)
        self.assertGreater(current.chat_level, 0.1)
        self.assertGreater(current.social_need, 0.1)


if __name__ == "__main__":
    unittest.main()
