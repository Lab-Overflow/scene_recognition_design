from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scorer import score_templates
from .types import Evidence, RouteTrace, SceneDecision, TemplateScore


@dataclass(slots=True)
class RouterConfig:
    confident_margin: float = 3.0
    unknown_threshold: float = 2.0
    top_k_candidates: int = 3


@dataclass(slots=True)
class RoutingResult:
    decision: SceneDecision
    ranked: list[TemplateScore]


class SceneRouter:
    def __init__(self, templates_path: str | Path, cfg: RouterConfig | None = None, vlm_client: Any = None):
        self.templates_path = Path(templates_path)
        self.templates: list[dict[str, Any]] = json.loads(self.templates_path.read_text(encoding="utf-8"))
        self.template_map = {template["id"]: template for template in self.templates}
        self.cfg = cfg or RouterConfig()
        self.vlm_client = vlm_client

    def rank(self, evidence: Evidence | dict[str, Any]) -> list[TemplateScore]:
        ev = evidence if isinstance(evidence, Evidence) else Evidence.from_dict(evidence)
        breakdowns = score_templates(self.templates, ev)

        ranked = []
        for item in breakdowns:
            template = self.template_map[item.template_id]
            ranked.append(TemplateScore(template_id=item.template_id, label=template["label"], score=item.final_score))
        return ranked

    def route(self, evidence: Evidence | dict[str, Any], frame_bytes: bytes | None = None) -> RoutingResult:
        ev = evidence if isinstance(evidence, Evidence) else Evidence.from_dict(evidence)
        ranked = self.rank(ev)
        if not ranked:
            trace = RouteTrace(top_scores=[], margin=0.0, decision="VLM_COLD", reason="No template configured")
            return RoutingResult(
                decision=SceneDecision(label="unknown", confidence=0.0, source="none", trace=trace),
                ranked=[],
            )

        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else TemplateScore(template_id="", label="", score=0.0)
        margin = top.score - second.score
        top_template = self.template_map[top.template_id]

        threshold_cfg = top_template.get("threshold", {})
        confident_threshold = float(threshold_cfg.get("confident", 8.0))
        borderline_threshold = float(threshold_cfg.get("borderline", max(self.cfg.unknown_threshold, confident_threshold * 0.55)))

        if top.score >= confident_threshold and margin >= self.cfg.confident_margin:
            confidence = min(0.99, top.score / max(confident_threshold, 1.0))
            trace = RouteTrace(
                top_scores=ranked[: self.cfg.top_k_candidates],
                margin=margin,
                decision="RULE_ACCEPT",
                reason=f"score={top.score:.2f} >= {confident_threshold:.2f} and margin={margin:.2f}",
            )
            return RoutingResult(
                decision=SceneDecision(label=top.label, confidence=confidence, source="rules", trace=trace),
                ranked=ranked,
            )

        candidates = ranked[: self.cfg.top_k_candidates]

        if top.score <= self.cfg.unknown_threshold:
            if self.vlm_client is None:
                trace = RouteTrace(
                    top_scores=candidates,
                    margin=margin,
                    decision="UNKNOWN",
                    reason=f"score={top.score:.2f} <= unknown_threshold={self.cfg.unknown_threshold:.2f}",
                )
                return RoutingResult(
                    decision=SceneDecision(label="unknown", confidence=0.0, source="rules", trace=trace),
                    ranked=ranked,
                )
            return self._route_by_vlm(ev, frame_bytes, candidates, margin, "VLM_COLD")

        if top.score >= borderline_threshold and self.vlm_client is None:
            confidence = min(0.8, top.score / max(confident_threshold, 1.0))
            trace = RouteTrace(
                top_scores=candidates,
                margin=margin,
                decision="RULE_CANDIDATE",
                reason=(
                    f"score in borderline range [{borderline_threshold:.2f}, {confident_threshold:.2f}), "
                    "no VLM configured"
                ),
            )
            return RoutingResult(
                decision=SceneDecision(label=top.label, confidence=confidence, source="rules_candidate", trace=trace),
                ranked=ranked,
            )

        if self.vlm_client is None:
            trace = RouteTrace(
                top_scores=candidates,
                margin=margin,
                decision="UNKNOWN",
                reason="No VLM configured and rule score not confident",
            )
            return RoutingResult(
                decision=SceneDecision(label="unknown", confidence=0.0, source="rules", trace=trace),
                ranked=ranked,
            )

        return self._route_by_vlm(ev, frame_bytes, candidates, margin, "VLM_CANDIDATE")

    def _route_by_vlm(
        self,
        evidence: Evidence,
        frame_bytes: bytes | None,
        candidates: list[TemplateScore],
        margin: float,
        decision: str,
    ) -> RoutingResult:
        vlm_res = self.vlm_client.classify(
            frame_bytes=frame_bytes,
            evidence=evidence,
            candidates=candidates,
            templates=self.templates,
        )
        label = str(vlm_res.get("label", "unknown"))
        confidence = float(vlm_res.get("confidence", 0.0))
        reason = str(vlm_res.get("reason", "vlm fallback"))

        trace = RouteTrace(top_scores=candidates, margin=margin, decision=decision, reason=reason)
        result = SceneDecision(label=label, confidence=confidence, source="vlm", trace=trace)
        return RoutingResult(decision=result, ranked=self.rank(evidence))
