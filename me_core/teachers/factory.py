from __future__ import annotations

import importlib
from typing import Dict, List

from .interface import DummyTeacher, Teacher, HumanTeacher
from .manager import TeacherManager


def create_teacher_manager_from_config(cfg: Dict[str, object] | None) -> TeacherManager:
    teachers: List[Teacher] = [DummyTeacher()]
    if cfg:
        use_real = bool(cfg.get("use_real_teacher", False))
        if use_real:
            module_name = cfg.get("teacher_module", "me_ext.teachers.real_teacher")
            kwargs = cfg.get("teacher_kwargs", {}) or {}
            try:
                module = importlib.import_module(str(module_name))
                factory = getattr(module, "create_teacher", None)
                if callable(factory):
                    real_teacher = factory(kwargs)
                    teachers.append(real_teacher)
            except Exception:
                # 回退仅使用 DummyTeacher
                pass
        if cfg.get("use_human_teacher"):
            mode = str(cfg.get("human_input_mode", "cli"))
            file_path = cfg.get("human_input_path")
            teachers.append(HumanTeacher(mode if mode in {"cli", "file"} else "cli", str(file_path) if file_path else None))
    return TeacherManager(teachers)


__all__ = ["create_teacher_manager_from_config"]
