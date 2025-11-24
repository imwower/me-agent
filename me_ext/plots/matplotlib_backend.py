from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt

from me_core.research.plot_types import PlotSpec


class PlotRenderer:
    """
    使用 matplotlib 将 PlotSpec 渲染为 PNG。
    """

    def __init__(self, output_dir: str = "reports/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, spec: PlotSpec) -> str:
        if spec.kind == "line":
            return self._render_line(spec)
        if spec.kind == "bar":
            return self._render_bar(spec)
        if spec.kind == "brain_graph":
            return self._render_brain_graph(spec)
        raise ValueError(f"未知图表类型: {spec.kind}")

    def _render_line(self, spec: PlotSpec) -> str:
        plt.figure()
        for series in spec.line_series:
            plt.plot(series.x, series.y, label=series.label)
        plt.title(spec.title)
        plt.xlabel(spec.x_label)
        plt.ylabel(spec.y_label)
        if len(spec.line_series) > 1:
            plt.legend()
        out = self.output_dir / f"{spec.id}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return str(out)

    def _render_bar(self, spec: PlotSpec) -> str:
        plt.figure()
        for idx, series in enumerate(spec.bar_series):
            positions = [i + idx * 0.2 for i in range(len(series.categories))]
            plt.bar(positions, series.values, width=0.2, label=series.label)
        plt.title(spec.title)
        plt.xlabel(spec.x_label)
        plt.ylabel(spec.y_label)
        plt.xticks(range(len(spec.bar_series[0].categories) if spec.bar_series else 0), spec.bar_series[0].categories if spec.bar_series else [])
        if len(spec.bar_series) > 1:
            plt.legend()
        out = self.output_dir / f"{spec.id}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return str(out)

    def _render_brain_graph(self, spec: PlotSpec) -> str:
        plt.figure()
        n = max(len(spec.graph_nodes), 1)
        positions = self._circle_positions(n, radius=1.0)
        node_pos = {node: positions[i] for i, node in enumerate(spec.graph_nodes)}
        for node, (x, y) in node_pos.items():
            plt.scatter(x, y, s=300, alpha=0.8)
            plt.text(x, y, node, ha="center", va="center", color="white", fontsize=9, weight="bold")
        for edge in spec.graph_edges:
            if edge.source not in node_pos or edge.target not in node_pos:
                continue
            x0, y0 = node_pos[edge.source]
            x1, y1 = node_pos[edge.target]
            plt.plot([x0, x1], [y0, y1], color="gray", alpha=0.6)
        plt.title(spec.title)
        plt.axis("off")
        out = self.output_dir / f"{spec.id}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return str(out)

    def _circle_positions(self, n: int, radius: float = 1.0) -> Tuple[Tuple[float, float], ...]:
        import math

        return tuple((radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n))
