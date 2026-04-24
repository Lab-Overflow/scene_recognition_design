from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_router import RouterConfig, SceneRouter, build_evidence
from scene_router.detector import YoloDetector
from scene_router.vlm import create_vlm_client


@dataclass(slots=True)
class RuntimeConfig:
    sample_interval: float
    smooth_window: int
    change_confirm: int


class CameraSource:
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera index={camera_index}")

        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Any:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame

    def close(self) -> None:
        self.cap.release()


class HttpFrameSource:
    def __init__(self, frame_url: str, timeout_s: float) -> None:
        import requests

        self.requests = requests
        self.url = frame_url
        self.timeout_s = timeout_s

    def read(self) -> Any:
        import cv2
        import numpy as np

        r = self.requests.get(self.url, timeout=self.timeout_s)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode JPEG from frame URL")
        return frame

    def close(self) -> None:
        return None


class SyntheticSource:
    def __init__(self, width: int, height: int) -> None:
        import numpy as np

        self.np = np
        self.width = max(320, int(width))
        self.height = max(240, int(height))
        self.index = 0

    def read(self) -> Any:
        import cv2

        self.index += 1
        frame = self.np.zeros((self.height, self.width, 3), dtype=self.np.uint8)
        # Gradient background for a deterministic smoke-test feed.
        for y in range(self.height):
            c = int(30 + 60 * y / max(1, self.height - 1))
            frame[y, :, :] = (c, c // 2, c + 20)
        cv2.putText(frame, "Synthetic Scene Feed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        cv2.putText(frame, f"frame={self.index}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 240, 180), 2)
        return frame

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live hybrid scene-recognition demo")
    parser.add_argument("--templates", default=str(ROOT / "config" / "scene_templates.json"))

    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--camera-index", type=int, default=0, help="Local camera index")
    src.add_argument("--frame-url", default="", help="HTTP single-frame endpoint (e.g., /frame.jpg)")
    src.add_argument("--synthetic", action="store_true", help="Use synthetic frame source for smoke tests")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Seconds between inferences")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV preview window")

    parser.add_argument("--detector", default="yolo", choices=["yolo", "none"])
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=640)

    parser.add_argument("--provider", default="none", choices=["none", "stub", "openai", "anthropic"])
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")

    parser.add_argument("--confident-margin", type=float, default=2.8)
    parser.add_argument("--unknown-threshold", type=float, default=2.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--change-confirm", type=int, default=3)
    parser.add_argument("--log-jsonl", default="", help="Optional output jsonl path")
    parser.add_argument("--state-file", default="logs/latest_state.json", help="Latest state JSON path for dashboard")
    return parser.parse_args()


def frame_to_jpeg(frame_bgr: Any, quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf)


def compute_global_cues(frame_bgr: Any, det_labels: list[str], prev_frame: Any | None) -> dict[str, float]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean() / 255.0)

    motion_score = 0.0
    if prev_frame is not None:
        prev_g = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_g)
        motion_score = float(np.mean(diff) / 255.0)

    outdoor_objects = {"car", "bus", "truck", "traffic_light", "bench", "tree", "bicycle", "motorcycle"}
    indoor_objects = {
        "monitor",
        "laptop",
        "keyboard",
        "bed",
        "sofa",
        "refrigerator",
        "oven",
        "microwave",
        "dining_table",
        "toilet",
        "sink",
    }

    label_set = set(det_labels)
    outdoor_hits = len(label_set & outdoor_objects)
    indoor_hits = len(label_set & indoor_objects)

    if indoor_hits > outdoor_hits:
        indoor_prob = 0.8
        outdoor_prob = 0.2
    elif outdoor_hits > indoor_hits:
        indoor_prob = 0.25
        outdoor_prob = 0.75
    else:
        indoor_prob = 0.5
        outdoor_prob = 0.5

    # Bright scenes with clear outdoor objects should increase outdoor confidence.
    if brightness > 0.65 and outdoor_hits > 0:
        outdoor_prob = min(0.9, outdoor_prob + 0.15)
        indoor_prob = max(0.1, 1.0 - outdoor_prob)

    return {
        "brightness": round(brightness, 4),
        "motion_score": round(motion_score, 4),
        "indoor_prob": round(indoor_prob, 4),
        "outdoor_prob": round(outdoor_prob, 4),
    }


def smooth_label(history: deque[tuple[str, float]]) -> tuple[str, float]:
    votes: Counter[str] = Counter()
    for label, score in history:
        votes[label] += float(score)
    if not votes:
        return "unknown", 0.0
    label, score = votes.most_common(1)[0]
    return label, float(score)


def open_source(args: argparse.Namespace) -> Any:
    if args.synthetic:
        return SyntheticSource(width=args.width, height=args.height)
    if args.frame_url:
        return HttpFrameSource(frame_url=args.frame_url, timeout_s=max(args.sample_interval, 1.0) + 2.0)
    return CameraSource(camera_index=args.camera_index, width=args.width, height=args.height)


def main() -> None:
    args = parse_args()
    runtime = RuntimeConfig(
        sample_interval=max(0.1, args.sample_interval),
        smooth_window=max(1, args.smooth_window),
        change_confirm=max(1, args.change_confirm),
    )

    source = open_source(args)

    detector = None
    if args.detector == "yolo":
        detector = YoloDetector(model_name=args.yolo_model, conf=args.det_conf, imgsz=args.det_imgsz)
    vlm_client = create_vlm_client(provider=args.provider, model=args.model or None, api_key=args.api_key or None)

    router = SceneRouter(
        templates_path=args.templates,
        cfg=RouterConfig(confident_margin=args.confident_margin, unknown_threshold=args.unknown_threshold, top_k_candidates=3),
        vlm_client=vlm_client,
    )

    preview_enabled = bool(args.preview and not args.frame_url)
    if args.preview and args.frame_url:
        print("[warn] --preview with --frame-url is disabled to avoid decode/display contention")

    history: deque[tuple[str, float]] = deque(maxlen=runtime.smooth_window)
    recent_labels: deque[str] = deque(maxlen=runtime.change_confirm)
    current_scene = "unknown"
    frame_no = 0
    prev_frame = None

    log_fp = None
    if args.log_jsonl:
        log_path = Path(args.log_jsonl)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", encoding="utf-8")
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    print("live demo started")
    print(
        json.dumps(
            {
                "source": "frame_url" if args.frame_url else f"camera:{args.camera_index}",
                "synthetic": bool(args.synthetic),
                "provider": args.provider,
                "model": args.model or "default",
                "detector": args.detector,
                "sample_interval": runtime.sample_interval,
                "preview": preview_enabled,
                "state_file": str(state_path),
            },
            ensure_ascii=False,
        )
    )

    try:
        while True:
            t0 = time.time()
            frame = source.read()
            frame_no += 1

            dets = detector.detect(frame) if detector is not None else []
            det_labels = [d["label"] for d in dets]
            cues = compute_global_cues(frame, det_labels, prev_frame)
            prev_frame = frame.copy()

            evidence = build_evidence(raw_detections=dets, global_cues=cues, min_score=args.det_conf)
            frame_jpeg = frame_to_jpeg(frame)
            routed = router.route(evidence=evidence, frame_bytes=frame_jpeg)

            raw_label = routed.decision.label
            raw_conf = routed.decision.confidence
            history.append((raw_label, raw_conf))
            smoothed_label, smoothed_score = smooth_label(history)

            recent_labels.append(smoothed_label)
            changed = False
            if len(recent_labels) == runtime.change_confirm and len(set(recent_labels)) == 1:
                if smoothed_label != current_scene:
                    current_scene = smoothed_label
                    changed = True

            top_scores = [
                {"id": s.template_id, "label": s.label, "score": round(float(s.score), 3)}
                for s in routed.ranked[:3]
            ]

            latency_ms = int((time.time() - t0) * 1000)
            line = {
                "ts": round(time.time(), 3),
                "frame": frame_no,
                "scene": current_scene,
                "raw_label": raw_label,
                "raw_conf": round(float(raw_conf), 3),
                "smoothed_label": smoothed_label,
                "smoothed_score": round(smoothed_score, 3),
                "source": routed.decision.source,
                "route_decision": routed.decision.trace.decision,
                "latency_ms": latency_ms,
                "changed": changed,
                "top3": top_scores,
                "counts": evidence.counts,
                "global": evidence.global_cues,
            }

            print(json.dumps(line, ensure_ascii=False))
            state_path.write_text(json.dumps(line, ensure_ascii=False), encoding="utf-8")
            if log_fp:
                log_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
                log_fp.flush()

            if preview_enabled:
                import cv2

                overlay = frame.copy()
                text = f"scene={current_scene} raw={raw_label}({raw_conf:.2f}) src={routed.decision.source}"
                cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("scene_demo", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames > 0 and frame_no >= args.max_frames:
                break

            elapsed = time.time() - t0
            if elapsed < runtime.sample_interval:
                time.sleep(runtime.sample_interval - elapsed)

    except KeyboardInterrupt:
        print("\nstopped by keyboard interrupt")
    finally:
        source.close()
        if log_fp:
            log_fp.close()
        if preview_enabled:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
