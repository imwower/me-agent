"""渲染关键 PlotSpec 为图片。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.workspace import Workspace
from me_core.tools import DumpBrainGraphTool
from me_core.brain.adapter import parse_brain_graph_from_json
from me_core.memory.log_index import LogIndex
from me_core.research.plot_builder import PlotBuilder
from me_ext.plots.matplotlib_backend import PlotRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染 PlotSpec")
    parser.add_argument("--workspace", type=str, default="configs/workspace.example.json")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/plots")
    args = parser.parse_args()

    ws = Workspace.from_json(args.workspace)
    renderer = PlotRenderer(args.output)
    specs = []

    # brain graph
    brain_repos = ws.get_brain_repos()
    if brain_repos:
        repo = brain_repos[0]
        gtool = DumpBrainGraphTool(ws)
        res = gtool.run({"repo_id": repo.id})
        if "summary" in res and res.get("metrics") is not None:
            try:
                graph = parse_brain_graph_from_json(repo.id, res.get("raw", res.get("summary")))
            except Exception:
                # 如果 summary 不是 JSON，尝试 repo.meta 的 structure_script
                rc, out, _ = ws.get_repo(repo.id).run_command(repo.meta.get("structure_script", ["python", "scripts/dump_brain_graph.py"]))
                if rc == 0:
                    graph = parse_brain_graph_from_json(repo.id, out)
                else:
                    graph = None
            if graph:
                specs.append(PlotBuilder.build_brain_graph_plot(graph))

    # experiment curve
    idx = LogIndex("logs")
    specs.append(PlotBuilder.build_experiment_curve_plot(idx, scenario_id=args.scenario))

    for spec in specs:
        path = renderer.render(spec)
        print(f"生成图表 {spec.id}: {path}")  # noqa: T201


if __name__ == "__main__":
    main()
