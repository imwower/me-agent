from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .interface import Teacher
from .types import PolicyPatch, TeacherInput, TeacherOutput


@dataclass
class TeacherManager:
    teachers: List[Teacher] = field(default_factory=list)

    def gather_advice(self, ti: TeacherInput) -> List[TeacherOutput]:
        outputs: List[TeacherOutput] = []
        for t in self.teachers:
            try:
                out = t.generate_advice(ti)
                outputs.append(out)
            except Exception:
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


__all__ = ["TeacherManager"]
