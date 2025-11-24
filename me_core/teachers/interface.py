from __future__ import annotations

from typing import Protocol

from .types import PolicyPatch, TeacherInput, TeacherOutput


class Teacher(Protocol):
    name: str

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        ...


class DummyTeacher:
    """简单规则的占位 Teacher。"""

    name = "dummy_teacher"

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        patches: list[PolicyPatch] = []
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
        if not advice_bits:
            advice_bits.append("保持现有策略，逐步累积经验。")
        return TeacherOutput(
            advice_text=" ".join(advice_bits),
            policy_patches=patches,
            meta={"teacher": self.name},
        )


__all__ = ["Teacher", "DummyTeacher"]
