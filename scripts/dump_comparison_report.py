"""生成配置/设计空间对比报告。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.memory.log_index import LogIndex
from me_core.research.comparison_builder import ComparisonBuilder
from me_core.research.plot_builder import PlotBuilder
from me_ext.plots.matplotlib_backend import PlotRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="生成对比报告")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/comparison")
    args = parser.parse_args()

    idx = LogIndex("logs")
    builder = ComparisonBuilder(idx)
    points = builder.build_config_points(scenario_filter=args.scenario, top_k=20)
    summary = builder.generate_text_summary(points)
    renderer = PlotRenderer()
    plot = PlotBuilder.build_experiment_curve_plot(idx, scenario_id=args.scenario)
    plot_path = renderer.render(plot)

    lines = ["# 对比报告", ""]
    lines.append(f"- 采样配置点数: {len(points)}")
    lines.append(f"- 摘要: {summary}")
    lines.append("")
    lines.append("## 配置点列表")
    for p in points:
        lines.append(f"- {p.id}: params={p.params}, metrics={p.metrics}")
    if plot_path:
        lines.append("")
        lines.append(f"![实验曲线]({plot_path})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_{int(time.time())}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"对比报告已写入 {out_path}")  # noqa: T201


if __name__ == "__main__":
    main()
