from __future__ import annotations

from typing import Protocol

from .types import ConfigPatch, PolicyPatch, TeacherInput, TeacherOutput


class Teacher(Protocol):
    name: str

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        ...


class DummyTeacher:
    """简单规则的占位 Teacher。"""

    name = "dummy_teacher"

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        patches: list[PolicyPatch] = []
        config_patches: list[ConfigPatch] = []
        notes = ti.notes or ""
        score_hint = ""
        if isinstance(ti.current_config, dict):
            score_hint = str(ti.current_config.get("last_score", ""))
        advice_bits: list[str] = []
        if score_hint:
            advice_bits.append(f"最近得分={score_hint}。")
        if ti.introspection and ti.introspection.mistakes:
            advice_bits.append("注意到一些错误，需要提高工具成功率或丰富模态。")
            patches.append(
                PolicyPatch(
                    target="drives",
                    path="curiosity.min_concept_count",
                    value=max(1, 2),
                    reason="触发好奇心以获取更多模态信息。",
                )
            )
        if "dev" in notes or "code" in notes:
            advice_bits.append("代码任务建议：先阅读相关文件，再保证单测通过。")
        if ti.experiment_results:
            # 简单规则：若存在 train_loss，则建议降低学习率
            for res in ti.experiment_results:
                loss_val = res.metrics.get("loss") or res.metrics.get("train_loss")
                if loss_val is not None and loss_val > 0.5:
                    config_patches.append(
                        ConfigPatch(
                            repo_id=res.step.repo_id,
                            config_path=ti.current_config.get("config_path", "configs/auto.json")
                            if isinstance(ti.current_config, dict)
                            else "configs/auto.json",
                            path="training.lr",
                            value=0.5,
                            reason="loss 较高，建议降低学习率。",
                        )
                    )
                    advice_bits.append("实验 loss 偏高，尝试降低学习率。")
                    break
        if getattr(ti, "brain_graph", None):
            advice_bits.append("注意脑结构指标，必要时稀疏化连接或调整区域规模。")
        if not advice_bits:
            advice_bits.append("保持现有策略，逐步累积经验。")
        return TeacherOutput(
            advice_text=" ".join(advice_bits),
            policy_patches=patches,
            config_patches=config_patches,
            meta={"teacher": self.name},
        )


__all__ = ["Teacher", "DummyTeacher"]
