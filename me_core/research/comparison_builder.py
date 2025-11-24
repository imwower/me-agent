from __future__ import annotations

from typing import Any, Dict, List, Optional

from me_core.memory.log_index import LogIndex
from me_core.research.comparison_types import ConfigPoint


class ComparisonBuilder:
    def __init__(self, log_index: LogIndex) -> None:
        self.log_index = log_index

    def build_config_points(self, scenario_filter: Optional[str] = None, top_k: int = 20) -> List[ConfigPoint]:
        raws = self.log_index.query(kinds=["experiment", "benchmark"], max_results=top_k)
        points: List[ConfigPoint] = []
        for r in raws:
            if scenario_filter and r.get("scenario_id") != scenario_filter:
                continue
            params = r.get("config", {}) if isinstance(r.get("config"), dict) else {}
            metrics: Dict[str, float] = {}
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
            points.append(ConfigPoint(id=str(r.get("id") or r.get("scenario_id") or len(points)), params=params, metrics=metrics))
        return points[:top_k]

    def generate_text_summary(self, points: List[ConfigPoint]) -> str:
        if not points:
            return "暂无可用对比点。"
        best_acc = None
        best_energy = None
        for p in points:
            acc = p.metrics.get("score") or p.metrics.get("acc")
            energy = p.metrics.get("energy")
            if acc is not None:
                if best_acc is None or acc > best_acc[1]:
                    best_acc = (p, acc)
            if energy is not None:
                if best_energy is None or energy < best_energy[1]:
                    best_energy = (p, energy)
        parts: List[str] = []
        if best_acc:
            parts.append(f"在精度指标上，配置 {best_acc[0].id} 表现最佳 (score={best_acc[1]:.3f})。")
        if best_energy:
            parts.append(f"在能耗指标上，配置 {best_energy[0].id} 更优 (energy={best_energy[1]:.3f})。")
        if not parts:
            parts.append("尚未发现显著差异，建议继续探索。")
        return " ".join(parts)
