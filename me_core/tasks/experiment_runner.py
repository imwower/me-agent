from __future__ import annotations

import json
import re
from typing import Dict, List

from me_core.workspace import Repo, Workspace

from .experiment_types import ExperimentResult, ExperimentScenario, ExperimentStep


def _parse_metrics(step: ExperimentStep, stdout: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if step.parse_mode == "plain":
        return metrics
    if step.parse_mode == "regex" and step.parse_pattern:
        m = re.search(step.parse_pattern, stdout)
        if not m:
            return metrics
        groups = m.groups()
        keys = step.metrics_keys or []
        if keys and groups:
            for name, g in zip(keys, groups):
                try:
                    metrics[name] = float(g)
                except Exception:
                    continue
        elif groups:
            try:
                metrics["value"] = float(groups[0])
            except Exception:
                pass
        return metrics
    if step.parse_mode == "json":
        try:
            obj = json.loads(stdout)
        except Exception:
            return metrics
        keys = step.metrics_keys
        if keys:
            for k in keys:
                v = obj.get(k)
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        else:
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        return metrics
    return metrics


def run_experiment_step(repo: Repo, step: ExperimentStep) -> ExperimentResult:
    rc, out, err = repo.run_command(step.command)
    metrics = _parse_metrics(step, out)
    return ExperimentResult(step=step, returncode=rc, stdout=out, stderr=err, metrics=metrics)


def run_experiment_scenario(workspace: Workspace, scenario: ExperimentScenario) -> List[ExperimentResult]:
    results: List[ExperimentResult] = []
    for step in scenario.steps:
        repo = workspace.get_repo(step.repo_id)
        res = run_experiment_step(repo, step)
        results.append(res)
    return results


def evaluate_experiment_results(results: List[ExperimentResult], formula: str) -> float:
    """
    将每个 step 的 metrics 合并后用一个简单的公式求分。
    - key 形式：<kind>_<metric>（如 train_loss / eval_acc）
    - 允许公式中直接使用这些 key；使用内置 eval，globals 禁止，locals 为数字字典。
    """

    values: Dict[str, float] = {}
    for res in results:
        prefix = res.step.kind
        for k, v in res.metrics.items():
            values[f"{prefix}_{k}"] = v
    if not values:
        return 0.0
    try:
        score = float(eval(formula, {"__builtins__": {}}, values))
    except Exception:
        score = 0.0
    return score


__all__ = ["run_experiment_step", "run_experiment_scenario", "evaluate_experiment_results"]
