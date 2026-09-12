"""Run both processes from one terminal.

Equivalent to running, in two terminals:
    python app.py
    python voice_agent.py --char 1 --port 8765

Usage:
    python main.py                      # defaults: --char 1 --port 8765
    python main.py --char 2 --port 8766
    python main.py --char 1 --char 3    # start an agent for several characters
    python main.py --generic-port 8766  # web app + generic (no-character) agent only
    python main.py --char 1 --generic-port 8766   # character agent + generic agent

This just launches the existing scripts as child processes; it does NOT
change any of your code. Logs from each are prefixed [web] / [agent:<id>]
so you can tell them apart. Ctrl+C stops everything cleanly.
"""
from __future__ import annotations
import argparse
import signal
import subprocess
import sys
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PY = sys.executable  # use the same interpreter / venv you launched this with


def _pump(proc: subprocess.Popen, prefix: str) -> None:
    """Read a child's combined stdout/stderr and reprint it with a prefix."""
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()


def _spawn(args: list[str], prefix: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [PY, "-u", *args],            # -u = unbuffered, so logs appear live
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=_pump, args=(proc, prefix), daemon=True).start()
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the web app + voice agent(s) together.")
    ap.add_argument("--char", type=int, action="append",
                    help="Character id for a voice agent (repeatable). Default: 1")
    ap.add_argument("--port", type=int, action="append",
                    help="Port for each agent, matched to --char order. Default: 8765")
    ap.add_argument("--host", default="0.0.0.0", help="Agent host (default 0.0.0.0)")
    ap.add_argument("--no-web", action="store_true", help="Don't start app.py (agents only)")
    ap.add_argument("--generic-port", type=int, default=None,
                    help="Also start a generic (no-character) voice agent on "
                         "this port, for Quick Chat's voice option "
                         "(e.g. --generic-port 8766). Doesn't touch the DB, "
                         "so it works even if no characters exist yet.")
    args = ap.parse_args()

    # Only default to character 1 if the user asked for character agents at
    # all (or asked for nothing in particular). If they explicitly only
    # want --generic-port, don't spawn a character agent that might point
    # at a character id that doesn't exist.
    if args.char is not None:
        chars = args.char
    elif args.generic_port is not None:
        chars = []
    else:
        chars = [1]
    ports = args.port or []
    # Fill in missing ports starting at 8765, incrementing per agent.
    while len(ports) < len(chars):
        ports.append(8765 + len(ports))

    procs: list[subprocess.Popen] = []

    if not args.no_web:
        procs.append(_spawn(["app.py"], "[web]   "))

    for char_id, port in zip(chars, ports):
        procs.append(_spawn(
            ["voice_agent.py", "--char", str(char_id),
             "--host", args.host, "--port", str(port)],
            f"[agent:{char_id}]",
        ))

    if args.generic_port:
        procs.append(_spawn(
            ["voice_agent.py", "--generic",
             "--host", args.host, "--port", str(args.generic_port)],
            "[agent:generic]",
        ))

    print("\n  Started:")
    if not args.no_web:
        print("    web app  -> http://localhost:5000")
    for char_id, port in zip(chars, ports):
        print(f"    agent    -> char {char_id} on ws://{args.host}:{port}")
    if args.generic_port:
        print(f"    agent    -> generic on ws://{args.host}:{args.generic_port}")
    print("\n  Press Ctrl+C to stop everything.\n")

    def shutdown(*_):
        print("\n  Shutting down...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # If any child dies on its own, bring the whole thing down.
    try:
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    print(f"\n  A process exited (code {code}) — stopping the rest.")
                    shutdown()
            for p in procs:
                try:
                    p.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
    except KeyboardInterrupt:
        shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
