from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .interface import Teacher
from .types import PolicyPatch, TeacherInput, TeacherOutput

logger = logging.getLogger(__name__)


@dataclass
class TeacherManager:
    teachers: List[Teacher] = field(default_factory=list)
    audit_log_path: Path = Path("logs/teacher_audit.jsonl")

    def gather_advice(self, ti: TeacherInput) -> List[TeacherOutput]:
        outputs: List[TeacherOutput] = []
        for t in self.teachers:
            try:
                out = t.generate_advice(ti)
                out.source_teacher_name = getattr(t, "name", "unknown")
                outputs.append(out)
                self._write_audit(out, ti)
            except Exception:
                logger.warning("Teacher %s 调用失败", getattr(t, "name", "unknown"))
                continue
        return outputs

    def aggregate_patches(self, outputs: List[TeacherOutput]) -> List[PolicyPatch]:
        aggregated: List[PolicyPatch] = []
        seen_paths: set[str] = set()
        for out in outputs:
            for patch in out.policy_patches:
                key = f"{patch.target}:{patch.path}"
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                aggregated.append(patch)
        return aggregated

    def _write_audit(self, output: TeacherOutput, ti: TeacherInput) -> None:
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.time(),
                "teacher": output.source_teacher_name,
                "scenario_id": ti.scenario_id,
                "policy_patches": len(output.policy_patches),
                "config_patches": len(output.config_patches),
                "meta": output.meta,
            }
            with self.audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("写入 teacher 审计日志失败")


__all__ = ["TeacherManager"]
