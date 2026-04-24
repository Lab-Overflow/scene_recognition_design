from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_router import RouterConfig, SceneRouter

TEMPLATES = ROOT / "config" / "scene_templates.json"


def _fake_hackathon_evidence() -> dict:
    return {
        "counts": {
            "person": 12,
            "laptop": 8,
            "monitor": 4,
            "keyboard": 5,
            "whiteboard": 1,
            "projector_screen": 1,
            "badge": 5,
            "banner": 1,
            "coffee_cup": 7,
            "backpack": 4,
        },
        "global": {
            "indoor_prob": 0.91,
            "outdoor_prob": 0.05,
            "brightness": 0.57,
        },
        "objects": [
            {"label": "laptop", "score": 0.95},
            {"label": "monitor", "score": 0.88},
            {"label": "whiteboard", "score": 0.77},
            {"label": "badge", "score": 0.74},
        ],
    }


def _fake_restaurant_evidence() -> dict:
    return {
        "counts": {
            "person": 6,
            "dining_table": 4,
            "plate": 8,
            "fork": 7,
            "knife": 7,
            "wine_glass": 5,
            "chair": 8,
        },
        "global": {
            "indoor_prob": 0.88,
            "outdoor_prob": 0.04,
            "brightness": 0.62,
        },
        "objects": [
            {"label": "dining_table", "score": 0.94},
            {"label": "wine_glass", "score": 0.9},
            {"label": "plate", "score": 0.92},
        ],
    }


def _fake_ambiguous_evidence() -> dict:
    return {
        "counts": {
            "person": 2,
            "chair": 3,
            "table": 1,
        },
        "global": {
            "indoor_prob": 0.52,
            "outdoor_prob": 0.44,
            "brightness": 0.4,
        },
        "objects": [
            {"label": "chair", "score": 0.65},
            {"label": "person", "score": 0.75},
        ],
    }


def run_case(name: str, evidence: dict, router: SceneRouter) -> None:
    result = router.route(evidence)
    print(f"\n=== {name} ===")
    print("decision:", json.dumps({
        "label": result.decision.label,
        "confidence": round(result.decision.confidence, 3),
        "source": result.decision.source,
        "route_decision": result.decision.trace.decision,
        "reason": result.decision.trace.reason,
    }, ensure_ascii=False))

    print("top_scores:")
    for item in result.ranked[:5]:
        print(f"  - {item.template_id:<22} {item.score:>6.2f}")


def main() -> None:
    router = SceneRouter(
        templates_path=TEMPLATES,
        cfg=RouterConfig(confident_margin=2.8, unknown_threshold=2.0, top_k_candidates=3),
    )

    run_case("hackathon_like", _fake_hackathon_evidence(), router)
    run_case("restaurant_like", _fake_restaurant_evidence(), router)
    run_case("ambiguous", _fake_ambiguous_evidence(), router)


if __name__ == "__main__":
    main()
