from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from me_core.brain.graph import BrainGraph
from me_core.memory.log_index import LogIndex
from me_core.population.types import AgentFitness
from .plot_types import PlotSpec, LineSeries, BarSeries, GraphEdge


class PlotBuilder:
    """
    仅构造 PlotSpec，不涉及具体绘图。
    """

    @staticmethod
    def build_brain_graph_plot(brain_graph: BrainGraph) -> PlotSpec:
        nodes = list(brain_graph.regions.keys())
        edges: List[GraphEdge] = []
        for conn in brain_graph.connections.values():
            w = None
            try:
                w = float(conn.sparsity) if hasattr(conn, "sparsity") else None
            except Exception:
                w = None
            edges.append(GraphEdge(source=conn.pre_region, target=conn.post_region, weight=w))
        meta: Dict[str, Any] = {
            "region_kinds": {r.id: r.kind for r in brain_graph.regions.values()},
            "title": brain_graph.meta.get("config_path") if hasattr(brain_graph, "meta") else "",
        }
        return PlotSpec(
            id="brain_graph",
            kind="brain_graph",
            title="BrainGraph 可视化",
            graph_nodes=nodes,
            graph_edges=edges,
            meta=meta,
        )

    @staticmethod
    def build_experiment_curve_plot(log_index: LogIndex, scenario_id: Optional[str] = None) -> PlotSpec:
        records = log_index.query(kinds=["experiment", "benchmark"], max_results=100)
        xs: List[float] = []
        ys: List[float] = []
        for i, r in enumerate(records):
            if scenario_id and r.get("scenario_id") != scenario_id:
                continue
            xs.append(float(i))
            ys.append(float(r.get("score", 0.0)))
        series = [LineSeries(label=scenario_id or "experiment", x=xs, y=ys)]
        return PlotSpec(
            id="experiment_curve",
            kind="line",
            title=f"实验曲线 {scenario_id or ''}",
            x_label="index",
            y_label="score",
            line_series=series,
        )

    @staticmethod
    def build_coevo_fitness_plot(fitness_list: List[AgentFitness]) -> PlotSpec:
        xs = list(range(len(fitness_list)))
        overall = [f.overall_score for f in fitness_list]
        lines = [LineSeries(label="overall_score", x=xs, y=overall)]
        if fitness_list and fitness_list[0].brain_metrics:
            brain_vals = [f.brain_metrics.get("energy", 0.0) for f in fitness_list]
            lines.append(LineSeries(label="brain_energy", x=xs, y=brain_vals))
        return PlotSpec(
            id=f"coevo_fitness_{uuid.uuid4()}",
            kind="line",
            title="CoEvo Fitness",
            x_label="generation",
            y_label="score",
            line_series=lines,
        )

    @staticmethod
    def build_bar_from_points(title: str, points: List[Dict[str, Any]], key: str) -> PlotSpec:
        cats = [str(i.get("id") or idx) for idx, i in enumerate(points)]
        vals = [float(i.get(key, 0.0)) for i in points]
        bars = [BarSeries(label=key, categories=cats, values=vals)]
        return PlotSpec(
            id=f"bar_{title}_{uuid.uuid4()}",
            kind="bar",
            title=title,
            x_label="id",
            y_label=key,
            bar_series=bars,
        )
