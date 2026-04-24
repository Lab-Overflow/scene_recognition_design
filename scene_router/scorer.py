from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import Evidence


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass(slots=True)
class ScoreBreakdown:
    template_id: str
    base_score: float
    confidence_factor: float
    boost_bonus: float
    final_score: float


def _parse_expr(expr: str) -> tuple[str, float]:
    expr = expr.strip()
    for op in (">=", "<=", "!=", "==", ">", "<"):
        if expr.startswith(op):
            return op, float(expr[len(op) :].strip())
    raise ValueError(f"Invalid expression: {expr}")


def _compare(value: float, op: str, target: float) -> bool:
    fn = _OPS.get(op)
    if fn is None:
        raise ValueError(f"Unsupported operator: {op}")
    return bool(fn(value, target))


def evaluate_condition(cond: dict[str, Any], evidence: Evidence) -> bool:
    op = cond.get("op", ">=")
    target = float(cond.get("value", 1))

    if "object" in cond:
        key = cond["object"]
        value = float(evidence.counts.get(key, 0))
        return _compare(value, op, target)

    if "global" in cond:
        key = cond["global"]
        value = float(evidence.global_cues.get(key, 0.0))
        return _compare(value, op, target)

    raise ValueError(f"Condition must contain 'object' or 'global': {cond}")


def evaluate_node(node: Any, evidence: Evidence) -> bool:
    if node is None:
        return True

    if isinstance(node, list):
        return all(evaluate_node(item, evidence) for item in node)

    if not isinstance(node, dict):
        raise ValueError(f"Condition node must be dict/list/None, got: {type(node)}")

    has_all = "all_of" in node
    has_any = "any_of" in node
    if has_all or has_any:
        ok_all = True
        ok_any = True
        if has_all:
            ok_all = all(evaluate_node(item, evidence) for item in (node.get("all_of") or []))
        if has_any:
            ok_any = any(evaluate_node(item, evidence) for item in (node.get("any_of") or []))
        return ok_all and ok_any

    return evaluate_condition(node, evidence)


def _mean_object_confidence(evidence: Evidence, labels: list[str]) -> float:
    if not evidence.objects:
        return 1.0

    scores: list[float] = []
    label_set = set(labels)
    for obj in evidence.objects:
        if obj.label in label_set:
            scores.append(float(obj.score))

    if not scores:
        return 1.0

    mean = sum(scores) / len(scores)
    # Keep confidence as a gentle multiplier, avoiding over-amplification.
    return min(1.2, max(0.6, mean))


def _evaluate_global_compat(global_cues: dict[str, Any], evidence: Evidence) -> bool:
    if not global_cues:
        return True

    for key, expr in global_cues.items():
        if isinstance(expr, str):
            op, target = _parse_expr(expr)
            value = float(evidence.global_cues.get(key, 0.0))
            if not _compare(value, op, target):
                return False
            continue

        if isinstance(expr, (int, float)):
            value = float(evidence.global_cues.get(key, 0.0))
            if value < float(expr):
                return False
            continue

        raise ValueError(f"Unsupported global cue expression for '{key}': {expr}")

    return True


def score_template(template: dict[str, Any], evidence: Evidence) -> ScoreBreakdown:
    template_id = template["id"]

    required = template.get("required")
    if required and not evaluate_node(required, evidence):
        return ScoreBreakdown(template_id, 0.0, 1.0, 0.0, 0.0)

    if not _evaluate_global_compat(template.get("global_cues", {}), evidence):
        return ScoreBreakdown(template_id, 0.0, 1.0, 0.0, 0.0)

    evidence_cfg = template.get("evidence", {})
    positive: dict[str, float] = evidence_cfg.get("positive", {})
    negative: dict[str, float] = evidence_cfg.get("negative", {})
    cap = int(template.get("max_count_per_object", 3))

    base_score = 0.0
    for label, weight in positive.items():
        base_score += float(weight) * min(int(evidence.counts.get(label, 0)), cap)

    for label, weight in negative.items():
        base_score += float(weight) * int(evidence.counts.get(label, 0))

    confidence_factor = _mean_object_confidence(evidence, list(positive.keys()))

    bonus = 0.0
    for boost in template.get("boost_conditions", []):
        node = boost.get("when")
        if evaluate_node(node, evidence):
            bonus += float(boost.get("bonus", 0.0))

    final = max(0.0, base_score * confidence_factor + bonus)
    return ScoreBreakdown(template_id, base_score, confidence_factor, bonus, final)


def score_templates(templates: list[dict[str, Any]], evidence: Evidence) -> list[ScoreBreakdown]:
    results = [score_template(template, evidence) for template in templates]
    return sorted(results, key=lambda item: item.final_score, reverse=True)
