"""Deterministic command handlers, transport-agnostic.

These are the "no agent in the loop" actions a user can invoke directly
from chat — slash commands in Discord, and (eventually) Slack equivalents.
Each function takes a workspace path plus the parsed parameters, runs the
underlying `herd` operation, and returns plain text suitable for posting
back to the channel.

No transport-specific code lives here; chat platforms handle their own
registration UI (slash command tree, etc.) and call into these handlers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_RUNNABLE = [sys.executable, "-m", "hyperherd.cli"]


# --- /status --------------------------------------------------------------

def cmd_status(workspace: Path) -> str:
    """Show sweep totals + per-trial table. Backed by `herd snapshot`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["snapshot", str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd snapshot` failed (exit {e.returncode}):\n{e.stderr or e.stdout}"

    try:
        snap = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return f"Unparseable snapshot output:\n{proc.stdout[:500]}"

    return _format_status(snap)


def _format_status(snap: dict) -> str:
    name = snap.get("sweep_name", "?")
    totals = snap.get("totals") or {}
    trials = snap.get("trials") or []

    order = ["ready", "submitted", "queued", "running",
             "completed", "failed", "cancelled"]
    counts = ", ".join(
        f"{totals[k]} {k}" for k in order if totals.get(k)
    ) or "(no trials yet)"
    counts += f" ({totals.get('total', len(trials))} total)"

    if not trials:
        return f"{name} — {counts}"

    rows = [("idx", "status", "elapsed", "name")]
    for t in trials:
        rows.append((
            str(t.get("index", "?")),
            (t.get("status") or "?")[:10],
            (t.get("elapsed") or "-")[:10],
            (t.get("experiment_name") or "-")[:40],
        ))

    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [
        " ".join(c.ljust(w) for c, w in zip(rows[0], widths)),
        " ".join("-" * w for w in widths),
    ]
    for r in rows[1:]:
        lines.append(" ".join(c.ljust(w) for c, w in zip(r, widths)))

    return f"{name} — {counts}\n\n" + "\n".join(lines)


# --- /stop ----------------------------------------------------------------

def cmd_stop(workspace: Path, index: int) -> str:
    """Cancel a single trial. Backed by `herd stop -i <index>`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["stop", "-i", str(index), str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd stop -i {index}` failed: {e.stderr or e.stdout}"
    out = (proc.stdout or "").strip() or "(no output)"
    return f"Stopped trial {index}.\n{out}"


def cmd_stop_all(workspace: Path) -> str:
    """Cancel every running/queued trial. Backed by `herd stop --all`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["stop", "--all", str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd stop --all` failed: {e.stderr or e.stdout}"
    out = (proc.stdout or "").strip() or "(no output)"
    return f"Stopped all live trials.\n{out}"


# --- /tail ----------------------------------------------------------------

def cmd_tail(workspace: Path, index: int, lines: int = 20) -> str:
    """Last N lines of trial <index>'s stderr log. Stdout is excluded —
    the action is almost always in stderr (Python tracebacks, SLURM
    notices, training prints from libs that write to stderr by default)."""
    if lines <= 0 or lines > 1000:
        return f"`lines` must be between 1 and 1000 (got {lines})."

    log_path = Path(workspace) / ".hyperherd" / "logs" / f"{index}.err"
    if not log_path.is_file():
        return f"No stderr log at `{log_path}` — trial {index} may not have started yet."

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return f"Couldn't read `{log_path}`: {e}"

    tail = "\n".join(content.splitlines()[-lines:])
    if not tail.strip():
        return f"Trial {index} stderr is empty (log file exists but has no content yet)."
    return f"`{log_path.name}` — last {lines} lines:\n{tail}"


# --- /help ----------------------------------------------------------------

def cmd_help() -> str:
    """List of available slash commands."""
    return (
        "**HerdDog commands**\n"
        "`/status` — sweep totals + per-trial table\n"
        "`/stop <index>` — cancel one trial\n"
        "`/stop_all` — cancel every live trial\n"
        "`/tail <index> [lines]` — last N lines of trial stderr (default 20)\n"
        "`/help` — this list\n"
        "\n"
        "For anything else (status questions, cadence changes, "
        "remediation policy), `@HerdDog` me — I'll wake the agent."
    )
