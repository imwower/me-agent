from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from me_core.memory.log_index import LogIndex
from me_core.research.notebook_types import ExperimentEntry, ExperimentNotebook
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel


def _to_entry(obj: Dict[str, Any]) -> ExperimentEntry:
    ts = float(obj.get("ts") or obj.get("time") or time.time())
    kind = str(obj.get("kind") or obj.get("mode") or "benchmark")
    desc = obj.get("scenario_id") or obj.get("description") or obj.get("id") or "experiment"
    metrics = {}
    for k, v in obj.items():
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    config_summary = obj.get("config", {}) if isinstance(obj.get("config"), dict) else {}
    notes = str(obj.get("notes") or obj.get("summary") or "")
    return ExperimentEntry(
        id=str(obj.get("id") or uuid.uuid4()),
        timestamp=ts,
        kind=kind if kind in {"benchmark", "devloop", "coevo", "brain_exp"} else "benchmark",
        description=str(desc),
        config_summary=config_summary,
        metrics=metrics,
        notes=notes,
    )


class NotebookBuilder:
    def __init__(self, log_index: LogIndex, world: Optional[SimpleWorldModel] = None, self_model: Optional[SimpleSelfModel] = None) -> None:
        self.log_index = log_index
        self.world = world
        self.self_model = self_model

    def build_notebook(
        self,
        kind_filters: Optional[List[str]] = None,
        time_window: Optional[Tuple[float, float]] = None,
        max_entries: int = 100,
    ) -> ExperimentNotebook:
        since, until = (None, None)
        if time_window:
            since, until = time_window
        raw = self.log_index.query(kinds=kind_filters, since=since, until=until, max_results=max_entries)
        entries = [_to_entry(r) for r in raw]
        title = "实验 Notebook"
        meta: Dict[str, Any] = {}
        if self.self_model is not None:
            meta["self"] = self.self_model.describe_self()  # type: ignore[arg-type]
        if self.world is not None:
            meta["world_summary"] = self.world.summarize()
        return ExperimentNotebook(id=str(uuid.uuid4()), title=title, entries=entries, meta=meta)
