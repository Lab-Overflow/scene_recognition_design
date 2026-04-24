from __future__ import annotations

from collections import Counter
from typing import Any

from .types import Detection, Evidence


# Alias normalization so templates can stay stable even if detector labels differ.
DEFAULT_ALIAS_MAP = {
    "tvmonitor": "monitor",
    "display": "monitor",
    "laptop computer": "laptop",
    "notebook": "laptop",
    "projector screen": "projector_screen",
    "white board": "whiteboard",
    "cup": "coffee_cup",
    "wineglass": "wine_glass",
    "server rack": "server_rack",
    "network switch": "network_switch",
    "traffic light": "traffic_light",
}


def normalize_label(label: str, alias_map: dict[str, str] | None = None) -> str:
    alias_map = alias_map or DEFAULT_ALIAS_MAP
    key = label.strip().lower().replace("-", " ")
    key = " ".join(key.split())
    return alias_map.get(key, key.replace(" ", "_"))


def build_evidence(
    raw_detections: list[dict[str, Any]],
    global_cues: dict[str, float] | None = None,
    min_score: float = 0.2,
    alias_map: dict[str, str] | None = None,
) -> Evidence:
    objects: list[Detection] = []
    counts = Counter()

    for item in raw_detections:
        score = float(item.get("score", 0.0))
        if score < min_score:
            continue

        label = normalize_label(str(item.get("label", "unknown")), alias_map=alias_map)
        det = Detection(
            label=label,
            score=score,
            bbox=item.get("bbox"),
            source=str(item.get("source", "detector")),
        )
        objects.append(det)
        counts[label] += 1

    return Evidence(
        objects=objects,
        counts=dict(counts),
        global_cues=global_cues or {},
    )
