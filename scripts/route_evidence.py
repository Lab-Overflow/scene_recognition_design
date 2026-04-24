from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_router import RouterConfig, SceneRouter
from scene_router.vlm import VLMStub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route one evidence JSON payload through templates")
    parser.add_argument("evidence", help="Path to evidence JSON")
    parser.add_argument(
        "--templates",
        default=str(ROOT / "config" / "scene_templates.json"),
        help="Template JSON path",
    )
    parser.add_argument("--use-vlm-stub", action="store_true", help="Enable VLM fallback stub")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k scores to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    vlm = VLMStub() if args.use_vlm_stub else None

    router = SceneRouter(
        templates_path=args.templates,
        cfg=RouterConfig(confident_margin=2.8, unknown_threshold=2.0, top_k_candidates=3),
        vlm_client=vlm,
    )
    result = router.route(evidence)

    payload = {
        "label": result.decision.label,
        "confidence": round(result.decision.confidence, 4),
        "source": result.decision.source,
        "route_decision": result.decision.trace.decision,
        "reason": result.decision.trace.reason,
        "top_scores": [
            {"template_id": item.template_id, "label": item.label, "score": round(item.score, 4)}
            for item in result.ranked[: args.top_k]
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
