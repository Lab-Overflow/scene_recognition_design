from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_router import RouterConfig, SceneRouter

TEMPLATES = ROOT / "config" / "scene_templates.json"


class RouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = SceneRouter(
            templates_path=TEMPLATES,
            cfg=RouterConfig(confident_margin=2.5, unknown_threshold=2.0, top_k_candidates=3),
        )

    def test_hackathon_rule_hits_without_vlm(self) -> None:
        evidence = {
            "counts": {
                "person": 10,
                "laptop": 7,
                "monitor": 4,
                "keyboard": 5,
                "whiteboard": 1,
                "projector_screen": 1,
                "badge": 4,
                "banner": 1,
            },
            "global": {"indoor_prob": 0.9, "brightness": 0.55},
            "objects": [
                {"label": "laptop", "score": 0.95},
                {"label": "monitor", "score": 0.9},
                {"label": "whiteboard", "score": 0.8},
            ],
        }
        result = self.router.route(evidence)

        self.assertEqual(result.decision.label, "hackathon")
        self.assertEqual(result.decision.source, "rules")
        self.assertEqual(result.decision.trace.decision, "RULE_ACCEPT")

    def test_very_low_evidence_returns_unknown(self) -> None:
        evidence = {
            "counts": {"person": 1},
            "global": {"indoor_prob": 0.5, "outdoor_prob": 0.5},
            "objects": [{"label": "person", "score": 0.7}],
        }
        result = self.router.route(evidence)

        self.assertEqual(result.decision.label, "unknown")
        self.assertIn(result.decision.trace.decision, {"UNKNOWN", "RULE_CANDIDATE"})


if __name__ == "__main__":
    unittest.main()
