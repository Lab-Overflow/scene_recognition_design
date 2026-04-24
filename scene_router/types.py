from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    bbox: list[float] | None = None
    source: str = "unknown"


@dataclass(slots=True)
class Evidence:
    ts: float | None = None
    image_hash: str | None = None
    resolution: list[int] | None = None
    objects: list[Detection] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    global_cues: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Evidence":
        objects = [Detection(**obj) for obj in payload.get("objects", [])]
        return cls(
            ts=payload.get("ts"),
            image_hash=payload.get("image_hash"),
            resolution=payload.get("resolution"),
            objects=objects,
            counts=payload.get("counts", {}),
            global_cues=payload.get("global", payload.get("global_cues", {})),
        )


@dataclass(slots=True)
class TemplateScore:
    template_id: str
    label: str
    score: float


@dataclass(slots=True)
class RouteTrace:
    top_scores: list[TemplateScore]
    margin: float
    decision: str
    reason: str


@dataclass(slots=True)
class SceneDecision:
    label: str
    confidence: float
    source: str
    trace: RouteTrace
