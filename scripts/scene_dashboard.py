from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple local dashboard for live scene output")
    parser.add_argument("--state-file", default="logs/latest_state.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def html_page() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Scene Dashboard</title>
  <style>
    :root { --bg:#0f172a; --card:#111827; --line:#334155; --text:#e5e7eb; --muted:#94a3b8; --ok:#22c55e; --warn:#f59e0b; }
    body { margin:0; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: radial-gradient(circle at 10% 10%, #1f2937, #0b1020 60%); color:var(--text); }
    .wrap { max-width:1100px; margin:24px auto; padding:0 16px; }
    .row { display:grid; grid-template-columns: 1.6fr 1fr; gap:16px; }
    .card { background:rgba(17,24,39,.88); border:1px solid var(--line); border-radius:12px; padding:16px; }
    .scene { font-size:42px; font-weight:800; letter-spacing:.2px; margin:4px 0 10px; color:var(--ok); }
    .muted { color:var(--muted); font-size:13px; }
    .kv { display:grid; grid-template-columns: 120px 1fr; gap:8px; margin:6px 0; }
    .bar { height:10px; background:#0b1220; border:1px solid var(--line); border-radius:99px; overflow:hidden; }
    .bar > i { display:block; height:100%; background:linear-gradient(90deg, #22c55e, #3b82f6); }
    .top-item { margin:10px 0; }
    pre { background:#0b1220; border:1px solid var(--line); border-radius:8px; padding:10px; max-height:360px; overflow:auto; font-size:12px; }
    .flag { display:inline-block; font-size:12px; padding:3px 8px; border-radius:999px; border:1px solid var(--line); margin-left:8px; }
    .changed { color:#f59e0b; border-color:#f59e0b66; }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"row\">
      <div class=\"card\">
        <div class=\"muted\">Current Scene</div>
        <div id=\"scene\" class=\"scene\">loading...</div>
        <div class=\"kv\"><div class=\"muted\">Raw</div><div id=\"raw\">-</div></div>
        <div class=\"kv\"><div class=\"muted\">Route</div><div id=\"route\">-</div></div>
        <div class=\"kv\"><div class=\"muted\">Source</div><div id=\"source\">-</div></div>
        <div class=\"kv\"><div class=\"muted\">Latency</div><div id=\"latency\">-</div></div>
        <div class=\"kv\"><div class=\"muted\">Frame</div><div id=\"frame\">-</div></div>
        <div class=\"kv\"><div class=\"muted\">Updated</div><div id=\"updated\">-</div></div>
      </div>
      <div class=\"card\">
        <div class=\"muted\">Top 3 Templates</div>
        <div id=\"top\"></div>
      </div>
    </div>
    <div class=\"card\" style=\"margin-top:16px\">
      <div class=\"muted\">Latest JSON</div>
      <pre id=\"rawjson\">{}</pre>
    </div>
  </div>
  <script>
    const fmt = (v) => (v === undefined || v === null || v === '') ? '-' : String(v);

    function render(state) {
      document.getElementById('scene').textContent = fmt(state.scene || state.smoothed_label || 'unknown');
      document.getElementById('raw').textContent = `${fmt(state.raw_label)} (${fmt(state.raw_conf)})`;
      document.getElementById('route').textContent = fmt(state.route_decision);
      document.getElementById('source').textContent = fmt(state.source);
      document.getElementById('latency').textContent = `${fmt(state.latency_ms)} ms`;
      document.getElementById('frame').textContent = fmt(state.frame);
      document.getElementById('updated').textContent = new Date().toLocaleTimeString();

      const top = Array.isArray(state.top3) ? state.top3 : [];
      const maxScore = Math.max(1, ...top.map(x => Number(x.score || 0)));
      const html = top.map((item) => {
        const sc = Number(item.score || 0);
        const pct = Math.max(0, Math.min(100, (sc / maxScore) * 100));
        return `
          <div class=\"top-item\">
            <div style=\"display:flex;justify-content:space-between;gap:8px\">
              <div>${item.label || item.id || 'unknown'}</div>
              <div class=\"muted\">${sc.toFixed(3)}</div>
            </div>
            <div class=\"bar\"><i style=\"width:${pct}%\"></i></div>
          </div>
        `;
      }).join('');
      document.getElementById('top').innerHTML = html || '<div class=\"muted\">No scores yet</div>';

      document.getElementById('rawjson').textContent = JSON.stringify(state, null, 2);
    }

    async function tick() {
      try {
        const res = await fetch('/api/state', { cache: 'no-store' });
        const data = await res.json();
        render(data || {});
      } catch (e) {
        console.error(e);
      }
    }

    setInterval(tick, 800);
    tick();
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    state_file: Path

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            body = html_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/state":
            payload: dict = {}
            try:
                if self.state_file.exists():
                    payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> int:
    args = parse_args()
    state_file = Path(args.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    Handler.state_file = state_file
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard listening on http://{args.host}:{args.port} (state_file={state_file})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
