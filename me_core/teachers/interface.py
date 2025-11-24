from __future__ import annotations

from typing import Protocol, Literal
import json
from pathlib import Path

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


class HumanTeacher:
    """人类在环 Teacher，可通过 CLI 或文件输入 JSON 建议。"""

    name = "human_teacher"

    def __init__(self, input_mode: Literal["cli", "file"] = "cli", file_path: str | None = None) -> None:
        self.input_mode = input_mode
        self.file_path = file_path

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        summary = {
            "scenario_id": ti.scenario_id,
            "notes": ti.notes,
            "brain_snapshot": getattr(ti, "brain_snapshot", None),
        }
        if self.input_mode == "file":
            path = Path(self.file_path or "human_teacher.json")
            tpl = {
                "advice_text": "请填写建议",
                "policy_patches": [],
                "config_patches": [],
            }
            if not path.exists() or not path.read_text(encoding="utf-8"):
                path.write_text(json.dumps({"summary": summary, "template": tpl}, ensure_ascii=False, indent=2), encoding="utf-8")
            # 若文件已存在且有内容，直接读取，避免阻塞测试
            if path.read_text(encoding="utf-8").strip():
                raw = path.read_text(encoding="utf-8")
            else:
                print(f"请编辑 {path} 后回车继续")  # noqa: T201
                input()
                raw = path.read_text(encoding="utf-8")
        else:
            print("=== HumanTeacher 摘要 ===")  # noqa: T201
            print(json.dumps(summary, ensure_ascii=False, indent=2))  # noqa: T201
            print("请输入 TeacherOutput JSON（回车跳过补丁）:")  # noqa: T201
            raw = input().strip()

        try:
            obj = json.loads(raw) if raw else {}
            advice = str(obj.get("advice_text") or obj.get("advice") or "human advice")
            patches = []
            for item in obj.get("policy_patches") or []:
                if not isinstance(item, dict) or "path" not in item or "value" not in item:
                    continue
                patches.append(
                    PolicyPatch(
                        target=str(item.get("target") or "drives"),
                        path=str(item.get("path") or ""),
                        value=item.get("value"),
                        reason=str(item.get("reason") or ""),
                    )
                )
            cfg_patches = []
            for item in obj.get("config_patches") or []:
                if not isinstance(item, dict) or "path" not in item or "value" not in item:
                    continue
                cfg_patches.append(
                    ConfigPatch(
                        repo_id=str(item.get("repo_id") or ""),
                        config_path=str(item.get("config_path") or ""),
                        path=str(item.get("path") or ""),
                        value=item.get("value"),
                        reason=str(item.get("reason") or ""),
                    )
                )
            return TeacherOutput(advice_text=advice, policy_patches=patches, config_patches=cfg_patches, meta={"human": True})
        except Exception:
            return TeacherOutput(advice_text=raw or "human advice", policy_patches=[], meta={"human": True})


__all__ = ["Teacher", "DummyTeacher", "HumanTeacher"]
