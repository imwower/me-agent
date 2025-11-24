from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List

from me_core.brain import BrainGraph
from me_core.introspection import IntrospectionLog
from .types import GeneratedTask, TaskTemplate


class TaskGenerator:
    """
    基于模板与历史表现的简易任务生成器。
    """

    def __init__(self, templates: List[TaskTemplate]) -> None:
        self.templates = templates

    def _choose_templates(self, gaps: List[str]) -> List[TaskTemplate]:
        if not gaps:
            return self.templates
        chosen: List[TaskTemplate] = []
        for tpl in self.templates:
            if tpl.kind in gaps:
                chosen.append(tpl)
        return chosen or self.templates

    def generate_tasks_from_gaps(
        self,
        introspections: List[IntrospectionLog] | List[Dict[str, Any]],
        benchmark_results: List[Dict[str, Any]],
        brain_graph: BrainGraph | None,
        max_new_tasks: int = 10,
    ) -> List[GeneratedTask]:
        gaps: List[str] = []
        for b in benchmark_results:
            score = float(b.get("score", 1.0))
            if score < 0.6:
                gaps.append(str(b.get("kind") or b.get("id") or "multimodal"))
        for it in introspections:
            mistakes = getattr(it, "mistakes", []) if hasattr(it, "mistakes") else (it.get("mistakes") if isinstance(it, dict) else [])
            if mistakes:
                gaps.append("codefix" if "code" in str(mistakes) else "multimodal")
        if brain_graph:
            # 若脑区域较少或能耗高，倾向 brain_memory 任务
            if len(brain_graph.regions) < 3:
                gaps.append("brain_memory")

        templates = self._choose_templates(gaps)
        tasks: List[GeneratedTask] = []
        for _ in range(max_new_tasks):
            tpl = random.choice(templates)
            payload = self._build_payload(tpl)
            task = GeneratedTask(
                id=str(uuid.uuid4()),
                template_id=tpl.id,
                payload=payload,
                expected_behavior=f"完成 {tpl.kind} 任务，遵循 {tpl.description}",
                labels={"keywords": payload.get("keywords", [])},
                difficulty=tpl.difficulty,
                kind=tpl.kind,
            )
            tasks.append(task)
        return tasks

    def _build_payload(self, tpl: TaskTemplate) -> Dict[str, Any]:
        if tpl.kind == "multimodal":
            return {
                "image_path": "data/benchmarks/images/dummy.png",
                "question": "图片里有什么？",
                "keywords": ["图片", "概念"],
            }
        if tpl.kind == "codefix":
            return {
                "repo_id": "codefix",
                "bug_description": "修复一个 off-by-one bug",
                "files": ["buggy.py"],
                "test_cmd": ["python", "-m", "pytest"],
            }
        if tpl.kind == "brain_memory":
            return {
                "delay": random.randint(3, 10),
                "noise": random.random(),
            }
        return {"note": "brain_control placeholder"}
