from __future__ import annotations

from typing import Dict, List, Sequence

from me_core.agent import SimpleAgent

from .types import Scenario, TaskResult, TaskStep


def _step_success(reply: str, expected_keywords: List[str], *, mode: str = "contains_any", case_insensitive: bool = True) -> float:
    if not expected_keywords:
        return 1.0 if reply else 0.0
    text = reply or ""
    if case_insensitive:
        text = text.lower()
        expected = [kw.lower() for kw in expected_keywords]
    else:
        expected = expected_keywords

    if mode == "not_contains":
        misses = sum(1 for kw in expected if kw and kw not in text)
        return 1.0 if misses == len(expected) else 0.0
    elif mode == "exact":
        return 1.0 if text.strip() == "".join(expected).strip() else 0.0
    else:  # contains_any/default
        hits = sum(1 for kw in expected if kw and kw in text)
        return hits / len(expected) if expected else 0.0


def run_scenario(agent: SimpleAgent, scenario: Scenario) -> TaskResult:
    """按顺序执行一个 Scenario，并用关键字匹配做简单评估。"""

    total_weight = 0.0
    total_score = 0.0
    per_step: List[Dict[str, object]] = []
    case_insensitive = bool(scenario.eval_config.get("case_insensitive", True))

    for step in scenario.steps:
        reply = agent.step(step.user_input, image_path=step.image_path)
        mode = step.eval_config.get("mode", "contains_any") if step.eval_config else "contains_any"
        score = _step_success(reply or "", step.expected_keywords or [], mode=mode, case_insensitive=case_insensitive)
        weight = step.weight
        total_weight += weight
        total_score += score * weight
        per_step.append(
            {
                "input": step.user_input,
                "image_path": step.image_path,
                "reply": reply,
                "score": score,
                "weight": weight,
            }
        )

    final_score = (total_score / total_weight) if total_weight > 0 else 0.0
    success = final_score >= 0.6
    return TaskResult(
        success=success,
        score=final_score,
        details={"steps": per_step, "scenario_id": scenario.id},
    )


def run_scenarios(agent: SimpleAgent, scenarios: Sequence[Scenario]) -> Dict[str, TaskResult]:
    results: Dict[str, TaskResult] = {}
    for sc in scenarios:
        results[sc.id] = run_scenario(agent, sc)
    return results


__all__ = ["run_scenario", "run_scenarios"]
