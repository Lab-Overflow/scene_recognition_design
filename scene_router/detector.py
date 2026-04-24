from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Map common detector labels to the template vocabulary.
YOLO_LABEL_MAP = {
    "tv": "monitor",
    "tvmonitor": "monitor",
    "cell phone": "phone",
    "dining table": "dining_table",
    "potted plant": "plant",
    "traffic light": "traffic_light",
    "wine glass": "wine_glass",
    "teddy bear": "toy",
    "sports ball": "ball",
}


def _normalize_det_label(label: str) -> str:
    key = label.strip().lower()
    if key in YOLO_LABEL_MAP:
        return YOLO_LABEL_MAP[key]
    return key.replace(" ", "_")


@dataclass(slots=True)
class YoloDetector:
    model_name: str = "yolov8n.pt"
    conf: float = 0.25
    imgsz: int = 640

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YoloDetector; install with `pip install ultralytics`"
            ) from exc

        self._model = YOLO(self.model_name)

    def detect(self, frame_bgr: Any) -> list[dict[str, Any]]:
        results = self._model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if not results:
            return []

        out: list[dict[str, Any]] = []
        r = results[0]
        names = getattr(r, "names", {})
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out

        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        for cid, c, bb in zip(cls, confs, xyxy):
            raw = names.get(int(cid), str(int(cid))) if isinstance(names, dict) else str(int(cid))
            out.append(
                {
                    "label": _normalize_det_label(raw),
                    "score": float(c),
                    "bbox": [float(v) for v in bb],
                    "source": "yolo",
                }
            )
        return out
