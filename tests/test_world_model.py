import logging
import unittest

from me_core.world_model.model_stub import SimpleEnvWorldModel

# 为测试输出配置基础日志，便于观察 world_model 更新过程
logging.basicConfig(level=logging.INFO)


class SimpleEnvWorldModelTestCase(unittest.TestCase):
    """简易环境世界模型桩的行为测试。"""

    def test_predict_and_update_and_uncertainty(self) -> None:
        """predict / update / estimate_uncertainty 的基本行为应合理。"""

        wm = SimpleEnvWorldModel()

        # 构造两条简单的转移：obs -> next_obs
        obs1 = [0.0, 0.0, 0.0]
        next1 = [0.0, 1.0, 0.0]
        obs2 = [0.0, 1.0, 0.0]
        next2 = [1.0, 1.0, 0.0]

        # 初次预测时模型尚无统计，预测应退化为“输出等于输入”
        pred_info = wm.predict(state={}, obs_embed=obs1, action="a")
        self.assertEqual(pred_info["pred_next_embed"], obs1)

        # update 后 last_error 应被更新，且为非负数
        avg_error = wm.update(
            [
                (obs1, "a", next1),
                (obs2, "b", next2),
            ]
        )
        self.assertGreaterEqual(avg_error, 0.0)
        self.assertAlmostEqual(wm.last_error, avg_error)

        # 再次预测应使用更新后的统计，误差估计源自 last_error
        pred_info2 = wm.predict(state={}, obs_embed=obs1, action="a")
        self.assertIn("pred_next_embed", pred_info2)
        self.assertIn("error", pred_info2)
        self.assertGreaterEqual(pred_info2["error"], 0.0)

        # 不确定性应落在 [0,1] 范围内
        unc = wm.estimate_uncertainty()
        self.assertGreaterEqual(unc, 0.0)
        self.assertLessEqual(unc, 1.0)


if __name__ == "__main__":
    unittest.main()

