"""生成实验 Notebook（Markdown）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.memory.log_index import LogIndex
from me_core.research.notebook_builder import NotebookBuilder
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel


def to_markdown(nb) -> str:
    lines = [f"# 实验 Notebook: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(nb.created_at))}", ""]
    lines.append(f"- 记录条数: {len(nb.entries)}")
    lines.append("")
    for idx, e in enumerate(nb.entries, 1):
        lines.append(f"## 条目 {idx}: [{e.kind}] {e.description}")
        lines.append(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e.timestamp))}")
        if e.config_summary:
            lines.append(f"- 配置摘要: `{json.dumps(e.config_summary, ensure_ascii=False)}`")
        if e.metrics:
            metrics_text = "; ".join(f"{k}={v:.3f}" for k, v in e.metrics.items())
            lines.append(f"- 指标: {metrics_text}")
        if e.notes:
            lines.append(f"- 注释: {e.notes}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成实验 Notebook")
    parser.add_argument("--kind", action="append", help="过滤 kind")
    parser.add_argument("--since", type=float, default=None)
    parser.add_argument("--until", type=float, default=None)
    parser.add_argument("--output", type=str, default="reports/notebooks")
    args = parser.parse_args()

    idx = LogIndex("logs")
    world = SimpleWorldModel()
    self_model = SimpleSelfModel()
    builder = NotebookBuilder(idx, world, self_model)
    nb = builder.build_notebook(kind_filters=args.kind, time_window=(args.since, args.until) if args.since or args.until else None)
    md = to_markdown(nb)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"notebook_{int(time.time())}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Notebook 已保存到 {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
