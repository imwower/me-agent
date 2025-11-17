from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from me_core.types import JsonDict

logger = logging.getLogger(__name__)

Embedding = List[float]


@dataclass
class SimpleEnvWorldModel:
    """用于环境自监督的简易世界模型桩实现。

    设计目标：
        - 不依赖任何外部库；
        - 提供 predict/update/estimate_uncertainty 三个接口；
        - 能够根据历史 (obs_embed, action, next_obs_embed) 估计
          下一步观测的“平均向量”，并计算预测误差。

    内部状态：
        - transition_stats:
            (obs_bin, action) -> (sum_next_embed, count)
          其中 obs_bin 是对 obs_embed 做粗略离散化后的元组；
        - last_error: 最近一次 update 的平均误差，用于估计不确定性。
    """

    transition_stats: Dict[Tuple[Tuple[int, ...], str], Tuple[List[float], int]] = field(
        default_factory=dict
    )
    last_error: float = 0.0

    def _bin_obs(self, obs_embed: Embedding, bins: int = 8) -> Tuple[int, ...]:
        """将连续向量粗略离散化到有限桶中，作为字典键。

        为了保持实现简单，这里使用:
            bin_i = int(x * bins) 截断到 [0, bins-1]
        """

        result: List[int] = []
        for x in obs_embed:
            v = x
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            idx = int(v * bins)
            if idx >= bins:
                idx = bins - 1
            result.append(idx)
        return tuple(result)

    # 公共接口 --------------------------------------------------------------

    def predict(
        self,
        state: JsonDict,
        obs_embed: Embedding,
        action: str,
    ) -> JsonDict:
        """基于历史统计，对下一步观测向量进行预测。

        参数：
            state: 外部传入的 world_model_state 摘要（当前实现未用到）；
            obs_embed: 当前观测的向量表示；
            action: 即将执行的动作名称。

        返回：
            {
                "pred_next_embed": 预测的下一步观测向量，
                "error": 对“预测误差”的估计（当前使用 last_error 兜底）
            }
        """

        _ = state  # 当前实现不使用外部 state，仅作为接口占位

        key = (self._bin_obs(obs_embed), action)
        stats = self.transition_stats.get(key)

        if stats is None:
            # 若无历史数据，则简单地假设“下一个观测与当前相同”
            pred = list(obs_embed)
        else:
            sum_vec, count = stats
            if count <= 0:
                pred = list(obs_embed)
            else:
                pred = [v / count for v in sum_vec]

        logger.info(
            "WorldModel.predict: action=%s, has_stats=%s",
            action,
            stats is not None,
        )

        return {
            "pred_next_embed": pred,
            "error": self.last_error,
        }

    def update(
        self,
        transitions: List[Tuple[Embedding, str, Embedding]],
    ) -> float:
        """根据一批 (obs_embed, action, next_obs_embed) 更新内部统计。

        返回：
            avg_error: 本批样本上预测误差的平均值（L1 范数）。
        """

        if not transitions:
            logger.info("WorldModel.update: 空的 transitions，跳过更新。")
            return self.last_error

        total_error = 0.0
        sample_count = 0

        for obs_embed, action, next_embed in transitions:
            key = (self._bin_obs(obs_embed), action)
            stats = self.transition_stats.get(key)

            # 计算“更新前”的预测误差，用于估计当前模型表现
            if stats is None:
                pred = list(obs_embed)
            else:
                sum_vec, count = stats
                if count <= 0:
                    pred = list(obs_embed)
                else:
                    pred = [v / count for v in sum_vec]

            # L1 距离作为误差度量
            err = sum(abs(a - b) for a, b in zip(pred, next_embed)) / max(
                len(next_embed), 1
            )
            total_error += err
            sample_count += 1

            # 使用该样本更新统计
            if stats is None:
                self.transition_stats[key] = (list(next_embed), 1)
            else:
                sum_vec, count = stats
                if len(sum_vec) < len(next_embed):
                    # 对齐长度，避免异常
                    sum_vec = sum_vec + [0.0] * (len(next_embed) - len(sum_vec))
                new_sum = [s + v for s, v in zip(sum_vec, next_embed)]
                self.transition_stats[key] = (new_sum, count + 1)

        avg_error = total_error / max(sample_count, 1)
        self.last_error = avg_error

        logger.info(
            "WorldModel.update: 样本数=%d, 平均误差=%.4f",
            sample_count,
            avg_error,
        )
        return avg_error

    def estimate_uncertainty(self, last_error: float | None = None) -> float:
        """根据最近预测误差估计不确定度，返回 [0,1]。

        简单策略：
            - 取传入的 last_error 或 self.last_error；
            - 假设误差 0.0 ≈ 完全确定，对应 0.0；
            - 误差 >= 1.0 视为高不确定，对应 1.0；
            - 中间线性插值，并裁剪到 [0,1]。
        """

        err = self.last_error if last_error is None else float(last_error)
        if err < 0.0:
            err = 0.0
        # 为防止极端值影响，将误差在 [0,2] 范围内线性压缩到 [0,1]
        if err >= 2.0:
            unc = 1.0
        else:
            unc = err / 2.0

        logger.info(
            "WorldModel.estimate_uncertainty: last_error=%.4f, uncertainty=%.4f",
            err,
            unc,
        )
        return unc

