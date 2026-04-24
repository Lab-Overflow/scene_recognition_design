from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

from .types import Evidence, TemplateScore


@dataclass(slots=True)
class VLMStub:
    """Fallback stub used during local development."""

    default_label: str = "unknown"

    def classify(
        self,
        frame_bytes: bytes | None,
        evidence: Evidence,
        candidates: list[TemplateScore],
        templates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if candidates:
            top = candidates[0]
            return {
                "label": top.label,
                "confidence": 0.65,
                "reason": "vlm_stub selected top rule candidate",
            }

        return {
            "label": self.default_label,
            "confidence": 0.2,
            "reason": "vlm_stub no candidate",
        }


def _extract_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    # Try plain JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try fenced or mixed text: find the first balanced object.
    start = text.find("{")
    if start < 0:
        return {}

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                frag = text[start : idx + 1]
                try:
                    parsed = json.loads(frag)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {}
    return {}


def _evidence_to_prompt_dict(evidence: Evidence) -> dict[str, Any]:
    counts = {k: int(v) for k, v in evidence.counts.items() if int(v) > 0}
    objects = [
        {"label": o.label, "score": round(float(o.score), 4)}
        for o in sorted(evidence.objects, key=lambda x: x.score, reverse=True)[:15]
    ]
    return {
        "counts": counts,
        "global": evidence.global_cues,
        "objects": objects,
    }


def _build_prompt(
    evidence: Evidence,
    candidates: list[TemplateScore],
    templates: list[dict[str, Any]],
) -> tuple[str, str]:
    allowed_labels = sorted({str(t["label"]) for t in templates})
    evidence_view = _evidence_to_prompt_dict(evidence)

    candidate_view = [
        {
            "template_id": c.template_id,
            "label": c.label,
            "score": round(float(c.score), 4),
        }
        for c in candidates
    ]

    system = (
        "You are a scene classifier. "
        "Return strict JSON with keys: label, confidence, reason. "
        "Confidence must be between 0 and 1."
    )

    user = (
        "Classify this frame into one scene label.\n"
        f"Allowed labels: {json.dumps(allowed_labels, ensure_ascii=False)}\n"
        f"Rule candidates: {json.dumps(candidate_view, ensure_ascii=False)}\n"
        f"Detected evidence: {json.dumps(evidence_view, ensure_ascii=False)}\n"
        "Output JSON only, e.g. "
        '{"label":"hackathon","confidence":0.78,"reason":"..."}'
    )
    return system, user


@dataclass(slots=True)
class OpenAIVLMClient:
    api_key: str
    model: str = "gpt-5.1"

    def classify(
        self,
        frame_bytes: bytes | None,
        evidence: Evidence,
        candidates: list[TemplateScore],
        templates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not frame_bytes:
            return {
                "label": candidates[0].label if candidates else "unknown",
                "confidence": 0.25,
                "reason": "no image passed to openai fallback",
            }

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAIVLMClient") from exc

        client = OpenAI(api_key=self.api_key)
        system, user = _build_prompt(evidence, candidates, templates)

        img_b64 = base64.b64encode(frame_bytes).decode("ascii")

        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        data = _extract_json_from_text(text)

        label = str(data.get("label") or (candidates[0].label if candidates else "unknown"))
        confidence = float(data.get("confidence", 0.5 if candidates else 0.25))
        reason = str(data.get("reason", "openai fallback"))
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "reason": reason,
        }


@dataclass(slots=True)
class AnthropicVLMClient:
    api_key: str
    model: str = "claude-opus-4-1"
    max_tokens: int = 300

    def classify(
        self,
        frame_bytes: bytes | None,
        evidence: Evidence,
        candidates: list[TemplateScore],
        templates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not frame_bytes:
            return {
                "label": candidates[0].label if candidates else "unknown",
                "confidence": 0.25,
                "reason": "no image passed to anthropic fallback",
            }

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for AnthropicVLMClient") from exc

        client = Anthropic(api_key=self.api_key)
        system, user = _build_prompt(evidence, candidates, templates)

        img_b64 = base64.b64encode(frame_bytes).decode("ascii")

        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": user},
                    ],
                }
            ],
        )

        text_chunks = []
        for block in msg.content:
            if getattr(block, "type", "") == "text":
                text_chunks.append(getattr(block, "text", ""))

        text = "\n".join(text_chunks).strip()
        data = _extract_json_from_text(text)

        label = str(data.get("label") or (candidates[0].label if candidates else "unknown"))
        confidence = float(data.get("confidence", 0.5 if candidates else 0.25))
        reason = str(data.get("reason", "anthropic fallback"))
        return {
            "label": label,
            "confidence": min(1.0, max(0.0, confidence)),
            "reason": reason,
        }


def create_vlm_client(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
) -> Any | None:
    p = provider.strip().lower()
    if p in {"none", "off", "disabled"}:
        return None

    if p == "stub":
        return VLMStub()

    if p == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        return OpenAIVLMClient(api_key=key, model=model or "gpt-5.1")

    if p in {"anthropic", "claude"}:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        return AnthropicVLMClient(api_key=key, model=model or "claude-opus-4-1")

    raise RuntimeError(f"Unsupported VLM provider: {provider}")
