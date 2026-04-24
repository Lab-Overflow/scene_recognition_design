from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_EXE = str((ROOT / ".venv" / "bin" / "python")) if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable


def load_local_env() -> None:
    for name in (".env", ".env.local"):
        path = ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def check_templates() -> tuple[bool, str]:
    path = ROOT / "config" / "scene_templates.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"templates parse failed: {exc}"
    if not isinstance(data, list) or not data:
        return False, "templates list is empty"
    has_hackathon = any(item.get("id") == "hackathon" for item in data if isinstance(item, dict))
    if not has_hackathon:
        return False, "hackathon template missing"
    return True, f"templates ok ({len(data)} templates)"


def check_provider(provider: str) -> tuple[bool, str]:
    provider = provider.lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY missing"
        return True, "OPENAI_API_KEY found"
    if provider in {"anthropic", "claude"}:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY missing"
        return True, "ANTHROPIC_API_KEY found"
    return True, "provider key check skipped"


def check_demo_router() -> tuple[bool, str]:
    p = run([PY_EXE, "scripts/demo_router.py"], timeout=45)
    if p.returncode != 0:
        return False, f"demo_router failed: {p.stderr.strip() or p.stdout.strip()}"
    out = p.stdout
    if "hackathon_like" not in out or '"label": "hackathon"' not in out:
        return False, "demo_router output does not contain hackathon hit"
    return True, "demo_router ok"


def check_unittests() -> tuple[bool, str]:
    p = run([PY_EXE, "-m", "unittest", "discover", "-s", "tests", "-q"], timeout=60)
    if p.returncode != 0:
        return False, f"unit tests failed: {p.stderr.strip() or p.stdout.strip()}"
    return True, p.stdout.strip() or "unit tests ok"


def check_live_smoke() -> tuple[bool, str]:
    p = run(
        [
            PY_EXE,
            "scripts/live_scene_demo.py",
            "--synthetic",
            "--detector",
            "none",
            "--provider",
            "stub",
            "--max-frames",
            "3",
            "--sample-interval",
            "0.05",
            "--state-file",
            "logs/latest_state.json",
        ],
        timeout=90,
    )
    if p.returncode != 0:
        return False, f"live smoke failed: {p.stderr.strip() or p.stdout.strip()}"
    if "live demo started" not in p.stdout:
        return False, "live smoke did not start correctly"
    state_file = ROOT / "logs" / "latest_state.json"
    if not state_file.exists():
        return False, "state file was not created"
    return True, "live smoke ok (synthetic source)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated local checks before live scene test")
    parser.add_argument("--provider", default="none", choices=["none", "openai", "anthropic"]) 
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_local_env()
    checks = [
        ("templates", check_templates),
        ("provider", lambda: check_provider(args.provider)),
        ("demo_router", check_demo_router),
        ("tests", check_unittests),
        ("live_smoke", check_live_smoke),
    ]

    failed = False
    for name, fn in checks:
        ok, msg = fn()
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if not ok:
            failed = True

    if failed:
        print("\nauto_debug result: FAILED")
        return 1

    print("\nauto_debug result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
