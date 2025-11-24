from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict
import sys

from me_core.brain import BrainGraph, BrainSnapshot, parse_brain_graph_from_json
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


@dataclass(slots=True)
class BrainInferTool(BaseTool):
    workspace: Workspace
    default_cfg: str = "configs/agency.yaml"
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="brain_infer",
            description="调用 brain/snn 仓库的在线推理脚本 run_brain_infer.py，返回 BrainSnapshot 摘要。",
            input_schema={
                "repo_id": "string",
                "task_id": "string",
                "text": "string",
                "features": "dict",
                "config_path": "string",
            },
            output_schema={"snapshot": "dict"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo_id = params.get("repo_id", "self-snn")
        task_id = params.get("task_id", "generic")
        text = params.get("text", "")
        features = params.get("features", {})
        cfg_path = params.get("config_path", self.default_cfg)

        repo = self.workspace.get_repo(repo_id)
        cmd = None
        if hasattr(repo, "spec") and getattr(repo.spec, "meta", None):
            cmd = repo.spec.meta.get("brain_infer_script")
        if not cmd:
            cmd = [sys.executable, "scripts/run_brain_infer.py"]

        args = [
            "--config",
            cfg_path,
            "--task-id",
            task_id,
            "--text",
            text,
            "--features",
            json.dumps(features, ensure_ascii=False),
        ]
        rc, out, err = repo.run_command(list(cmd) + args)
        if rc != 0:
            return {"error": err or "brain_infer_failed"}
        try:
            data = json.loads(out)
        except Exception:
            return {"error": "invalid_json", "raw": out}

        snapshot = BrainSnapshot(
            repo_id=repo_id,
            region_activity=data.get("region_activity", {}) or {},
            global_metrics=data.get("global_metrics", {}) or {},
            memory_summary=data.get("memory_summary", {}) or {},
            decision_hint=data.get("decision_hint", {}) or {},
        )
        return {
            "snapshot": {
                "repo_id": snapshot.repo_id,
                "region_activity": snapshot.region_activity,
                "global_metrics": snapshot.global_metrics,
                "memory_summary": snapshot.memory_summary,
                "decision_hint": snapshot.decision_hint,
                "created_at": snapshot.created_at,
            }
        }


__all__ = ["DumpBrainGraphTool", "EvalBrainEnergyTool", "EvalBrainMemoryTool", "BrainInferTool"]
