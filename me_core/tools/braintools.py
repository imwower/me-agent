from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

from me_core.brain import BrainGraph, parse_brain_graph_from_json
from me_core.tools.base import BaseTool, ToolSpec
from me_core.workspace import Workspace


@dataclass(slots=True)
class DumpBrainGraphTool:
    workspace: Workspace
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="dump_brain_graph",
            description="调用 brain 仓库的结构脚本，解析脑图谱。",
            input_schema={"repo_id": "string"},
            output_schema={"summary": "string", "metrics": "list"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        script_cmd = repo.spec.meta.get("structure_script") if hasattr(repo, "spec") else None
        if not script_cmd:
            return {"summary": "no structure_script", "metrics": []}
        rc, out, err = repo.run_command(script_cmd)
        if rc != 0:
            return {"summary": f"structure_script failed: {err}", "metrics": []}
        graph: BrainGraph = parse_brain_graph_from_json(repo.spec.id, out)
        metrics = [{"name": m.name, "value": m.value, "unit": m.unit} for m in graph.metrics]
        return {"summary": graph.summary(), "metrics": metrics}


@dataclass(slots=True)
class EvalBrainEnergyTool:
    workspace: Workspace
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="eval_brain_energy",
            description="运行 energy 脚本，返回能耗指标。",
            input_schema={"repo_id": "string"},
            output_schema={"energy": "float", "unit": "string", "raw": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        cmd = repo.spec.meta.get("energy_script") if hasattr(repo, "spec") else None
        if not cmd:
            return {"energy": 0.0, "unit": "", "raw": ""}
        rc, out, err = repo.run_command(cmd)
        energy = 0.0
        if rc == 0:
            try:
                obj = json.loads(out)
                energy = float(obj.get("energy", 0.0))
                unit = str(obj.get("unit", ""))
            except Exception:
                unit = ""
        else:
            unit = ""
        return {"energy": energy, "unit": unit, "raw": out + err}


@dataclass(slots=True)
class EvalBrainMemoryTool:
    workspace: Workspace
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="eval_brain_memory",
            description="运行记忆评估脚本。",
            input_schema={"repo_id": "string"},
            output_schema={"capacity": "float", "unit": "string", "raw": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        cmd = repo.spec.meta.get("memory_script") if hasattr(repo, "spec") else None
        if not cmd:
            return {"capacity": 0.0, "unit": "", "raw": ""}
        rc, out, err = repo.run_command(cmd)
        capacity = 0.0
        unit = ""
        if rc == 0:
            try:
                obj = json.loads(out)
                capacity = float(obj.get("capacity", 0.0))
                unit = str(obj.get("unit", ""))
            except Exception:
                unit = ""
        return {"capacity": capacity, "unit": unit, "raw": out + err}


__all__ = ["DumpBrainGraphTool", "EvalBrainEnergyTool", "EvalBrainMemoryTool"]
