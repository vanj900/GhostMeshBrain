#!/usr/bin/env python3
"""dashboard.py — near-real-time vitals dashboard for GhostMesh.

Reads GHOST_VITALS_LOG (JSONL) and streams the latest vitals to the terminal
using curses (default) or a simple HTTP JSON endpoint.

Usage
-----
Curses mode (default):
    GHOST_VITALS_LOG=/tmp/ghost_vitals.jsonl python scripts/dashboard.py

HTTP mode (streams JSON from /vitals):
    python scripts/dashboard.py --mode http --port 8765

Then open http://localhost:8765/vitals in your browser.

Both modes auto-refresh when new records are appended to the JSONL file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Shared vitals reader ──────────────────────────────────────────────────────

def _read_latest_vitals(path: str, n: int = 1) -> list[dict]:
    """Return the last *n* vitals records from the JSONL log.

    Parameters
    ----------
    path:
        Path to the GHOST_VITALS_LOG JSONL file.
    n:
        Number of most-recent records to return (default 1).
    """
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("event") == "respawn":
                        continue
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records[-n:] if records else []


def _fmt_float(v, fmt: str = ".2f") -> str:
    if v is None:
        return "  n/a"
    return f"{float(v):{fmt}}"


# ── Curses dashboard ──────────────────────────────────────────────────────────

def _run_curses(vitals_path: str, refresh_hz: float = 2.0) -> None:
    """Run the curses dashboard.  Press 'q' to quit."""
    import curses

    def _draw(stdscr: "curses._CursesWindow") -> None:  # type: ignore[name-defined]
        curses.curs_set(0)
        stdscr.nodelay(True)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

        interval = 1.0 / max(refresh_hz, 0.1)
        last_file_size: int = -1

        while True:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                break

            # Only re-read file when it grows
            try:
                file_size = os.path.getsize(vitals_path)
            except OSError:
                file_size = 0

            if file_size != last_file_size:
                last_file_size = file_size
                records = _read_latest_vitals(vitals_path, n=50)
            else:
                records = _read_latest_vitals(vitals_path, n=50) if file_size else []

            latest = records[-1] if records else {}
            height, width = stdscr.getmaxyx()
            stdscr.clear()

            def _addstr(y: int, x: int, text: str, attr: int = 0) -> None:
                try:
                    stdscr.addstr(y, x, text[:width - x], attr)
                except curses.error:
                    pass

            def _health_attr(value: float, low_bad: bool = True) -> int:
                if low_bad:
                    if value >= 70:
                        return curses.color_pair(1)
                    if value >= 35:
                        return curses.color_pair(2)
                    return curses.color_pair(3)
                else:
                    if value <= 30:
                        return curses.color_pair(1)
                    if value <= 65:
                        return curses.color_pair(2)
                    return curses.color_pair(3)

            row = 0
            _addstr(row, 0, "GhostMesh Live Dashboard  [q = quit]",
                    curses.color_pair(4) | curses.A_BOLD)
            row += 1
            _addstr(row, 0, f"  Log: {vitals_path}")
            row += 1
            _addstr(row, 0, "─" * min(60, width - 1), curses.color_pair(4))
            row += 1

            if not latest:
                _addstr(row, 0, "  Waiting for vitals data…", curses.color_pair(2))
            else:
                tick = latest.get("tick", 0)
                mask = latest.get("mask", "—")
                stage = latest.get("stage", "—")
                _addstr(row, 0, f"  Tick: {tick:>8}   Stage: {stage:<10}  Mask: {mask:<14}")
                row += 1

                vitals = [
                    ("Energy   (E)", latest.get("energy", 0.0), True),
                    ("Heat     (T)", latest.get("heat", 0.0), False),
                    ("Waste    (W)", latest.get("waste", 0.0), False),
                    ("Integrity(M)", latest.get("integrity", 0.0), True),
                    ("Stability(S)", latest.get("stability", 0.0), True),
                ]
                for label, val, low_bad in vitals:
                    fval = float(val)
                    bar_w = 20
                    filled = int(round(fval / 100.0 * bar_w))
                    bar = "█" * max(0, min(bar_w, filled)) + "░" * (bar_w - max(0, min(bar_w, filled)))
                    attr = _health_attr(fval, low_bad=low_bad)
                    _addstr(row, 0, f"  {label:<14} ", 0)
                    _addstr(row, 18, bar, attr)
                    _addstr(row, 40, f" {fval:6.1f}", attr)
                    row += 1

                row += 1
                # Allostatic load
                al = float(latest.get("allostatic_load", 0.0))
                _addstr(row, 0, f"  Allostatic Load : {al:6.1f}",
                        _health_attr(al, low_bad=False))
                row += 1

                # Bifurcation status
                pi = latest.get("plasticity_index")
                pcs = latest.get("pre_collapse_score")
                tsc = latest.get("ticks_since_calcification")
                n_awakenings = latest.get("awakening_count", 0)
                if pi is not None:
                    pi = float(pi)
                    if pi >= 0.5:
                        bif_attr = curses.color_pair(1)
                        regime = "PLASTIC"
                    elif pi >= 0.3:
                        bif_attr = curses.color_pair(2)
                        regime = "TRANSITIONING"
                    else:
                        bif_attr = curses.color_pair(3)
                        regime = "GUARDIAN LOCK"
                    _addstr(row, 0, "─" * min(60, width - 1), curses.color_pair(4))
                    row += 1
                    _addstr(row, 0, "  Bifurcation Status", curses.A_BOLD)
                    row += 1
                    _addstr(row, 0, f"  Plasticity     : {pi:.3f}  [{regime}]", bif_attr)
                    row += 1
                    _addstr(row, 0, f"  Pre-Collapse   : {_fmt_float(pcs, '.3f')}", curses.color_pair(2))
                    row += 1
                    tsc_str = str(tsc) if tsc is not None else "n/a"
                    _addstr(row, 0, f"  Calcification  : {tsc_str:>6} ticks ago")
                    row += 1
                    awakening_attr = curses.color_pair(3) if n_awakenings > 0 else 0
                    _addstr(row, 0, f"  Awakenings     : {n_awakenings:>4}", awakening_attr)
                    row += 1

            _addstr(height - 1, 0,
                    f"  Refreshing at {refresh_hz:.1f} Hz — press 'q' to quit",
                    curses.color_pair(4) | curses.A_DIM)

            stdscr.refresh()
            time.sleep(interval)

    curses.wrapper(_draw)


# ── HTTP dashboard ────────────────────────────────────────────────────────────

def _run_http(vitals_path: str, port: int = 8765) -> None:
    """Serve latest vitals as JSON at http://localhost:<port>/vitals."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    _path = vitals_path  # captured in closure

    class VitalsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/vitals":
                records = _read_latest_vitals(_path, n=10)
                body = json.dumps(records, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ("/", "/index.html"):
                html = _make_html_page(port)
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # suppress request logging

    server = HTTPServer(("", port), VitalsHandler)
    print(f"[GhostMesh Dashboard] Serving at http://localhost:{port}/")
    print(f"  Open in browser or run: curl http://localhost:{port}/vitals")
    print("  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def _make_html_page(port: int) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GhostMesh Dashboard</title>
<style>
  body {{background:#111;color:#ccc;font-family:monospace;padding:20px}}
  h1 {{color:#0ff}} pre {{background:#1a1a1a;padding:16px;border-radius:6px;overflow:auto}}
  .green {{color:#0f0}} .yellow {{color:#ff0}} .red {{color:#f44}}
</style>
<script>
async function refresh() {{
  try {{
    const r = await fetch('/vitals');
    const data = await r.json();
    document.getElementById('data').textContent = JSON.stringify(data[data.length-1]||{{}},null,2);
  }} catch(e) {{
    document.getElementById('data').textContent = 'Waiting for data...';
  }}
}}
setInterval(refresh, 1000);
refresh();
</script>
</head>
<body>
<h1>GhostMesh Live Vitals</h1>
<p>Auto-refreshes every second. Raw endpoint: <a href="/vitals" style="color:#0ff">/vitals</a></p>
<pre id="data">Loading...</pre>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GhostMesh near-real-time vitals dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log",
        default=os.environ.get("GHOST_VITALS_LOG", ""),
        help="Path to GHOST_VITALS_LOG JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["curses", "http"],
        default="curses",
        help="Dashboard mode: curses (terminal) or http (browser)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP server port (only used in http mode)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=2.0,
        help="Refresh rate in Hz (curses mode only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.log:
        print(
            "Error: No vitals log path specified.\n"
            "  Set GHOST_VITALS_LOG or pass --log <path>",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "http":
        _run_http(args.log, port=args.port)
    else:
        _run_curses(args.log, refresh_hz=args.hz)
