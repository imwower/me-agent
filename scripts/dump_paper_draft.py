"""生成论文/技术报告草稿（Markdown/可选 LaTeX）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.memory.log_index import LogIndex
from me_core.research.notebook_builder import NotebookBuilder
from me_core.research.comparison_builder import ComparisonBuilder
from me_core.research.paper_builder import PaperDraftBuilder
from me_core.teachers.manager import TeacherManager
from me_core.teachers.interface import DummyTeacher
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel


def _markdown_from_draft(draft) -> str:
    lines = [f"# {draft.title}", "", "## Abstract", draft.abstract, ""]
    for sec in draft.sections:
        lines.append(f"## {sec.title}")
        lines.append(sec.content)
        for sub in sec.subsections:
            lines.append(f"### {sub.title}")
            lines.append(sub.content)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成论文/报告草稿")
    parser.add_argument("--output", type=str, default="reports/paper_draft")
    args = parser.parse_args()

    idx = LogIndex("logs")
    world = SimpleWorldModel()
    self_model = SimpleSelfModel()
    nb_builder = NotebookBuilder(idx, world, self_model)
    comp_builder = ComparisonBuilder(idx)
    draft_builder = PaperDraftBuilder(nb_builder, comp_builder, TeacherManager([DummyTeacher()]))
    draft = draft_builder.build_draft_outline()
    md = _markdown_from_draft(draft)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"paper_{int(time.time())}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"草稿已写入 {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
