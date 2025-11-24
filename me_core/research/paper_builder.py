from __future__ import annotations

import uuid
from typing import List

from me_core.research.paper_types import PaperDraft, Section
from me_core.research.notebook_builder import NotebookBuilder
from me_core.research.comparison_builder import ComparisonBuilder
from me_core.teachers.manager import TeacherManager
from me_core.teachers.interface import DummyTeacher


class PaperDraftBuilder:
    def __init__(self, notebook_builder: NotebookBuilder, comparison_builder: ComparisonBuilder, teacher_manager: TeacherManager | None = None) -> None:
        self.notebook_builder = notebook_builder
        self.comparison_builder = comparison_builder
        self.teacher_manager = teacher_manager or TeacherManager([DummyTeacher()])

    def build_draft_outline(self) -> PaperDraft:
        notebook = self.notebook_builder.build_notebook(kind_filters=None, max_entries=5)
        points = self.comparison_builder.build_config_points(top_k=5)
        summary_text = self.comparison_builder.generate_text_summary(points)
        abstract = (
            f"我们在多模态、代码修复与脑启发任务上进行了 {len(notebook.entries)} 次实验，"
            f"初步对比表明：{summary_text}"
        )
        sections: List[Section] = [
            Section(
                title="Introduction",
                content="背景、目标与主要贡献的概述。",
                subsections=[
                    Section(title="背景", content="讨论认知架构、自我驱动 agent 与脑启发 SNN 的动机。"),
                    Section(title="目标", content="让 agent 能自生成任务、联合 self-snn 进化，并形成研究资产。"),
                    Section(title="贡献", content="提供自动任务生成、课程表、联合进化与报告生成。"),
                ],
            ),
            Section(
                title="System Overview",
                content="描述 me-agent + self-snn 架构：感知/世界/自我/脑工具/DevLoop/CoEvo。",
            ),
            Section(
                title="Methods",
                content="介绍任务生成器、任务池与 curriculum、TrainSchedule、自演化流程。",
            ),
            Section(
                title="Experiments",
                content="基准、联合进化、brain 模式的实验设计与指标。总结：" + summary_text,
            ),
            Section(
                title="Discussion & Future Work",
                content="局限：规则化生成/评估较粗糙；未来可接入更强的 LLM/真实数据集与可视化。",
            ),
        ]
        return PaperDraft(title=f"Self-SNN Co-Evolution Report {uuid.uuid4()}", abstract=abstract, sections=sections, meta={"points": len(points)})
