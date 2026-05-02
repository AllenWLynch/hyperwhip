"""Deterministic per-tick state assembler.

Runs in pure Python before the agent is invoked. Pulls a fresh `herd
snapshot`, rotates the previous-tick snapshot for delta detection, drains
the inbox of any user messages received since last tick, and packages it
all into a single `TickState` the agent's `read_state()` tool returns.

The agent never calls `herd snapshot`, never reads the manifest, never
diffs anything itself — that's the point.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


SNAPSHOT_FILE = "last-snapshot.json"
PREV_SNAPSHOT_FILE = "last-snapshot.prev.json"
INBOX_FILE = "inbox.jsonl"
PLAN_FILE = "MONITOR_PLAN.md"


TickTrigger = Literal["scheduled", "failure", "completion", "user_message", "boot"]


@dataclass
class FailureView:
    """Diff entry: a trial that newly entered `failed` since the last snapshot."""
    index: int
    experiment_name: Optional[str]
    slurm_state: Optional[str]
    stderr_tail: List[str] = field(default_factory=list)


@dataclass
class InboundMessage:
    """A Discord (or other source) message received since the last tick."""
    timestamp: str       # ISO-8601 UTC
    source: str          # "discord" / "slack" / etc.
    author: str
    text: str


@dataclass
class TickState:
    """The single document the agent reads at the start of every tick."""
    sweep_name: str
    workspace: str                      # absolute path
    trigger: TickTrigger
    plan: str                           # MONITOR_PLAN.md contents (empty on first tick)
    totals: Dict[str, int]              # status -> count, plus "total"
    trials: List[Dict[str, Any]]        # straight from `herd snapshot`'s trials[]
    newly_failed: List[FailureView]
    newly_completed: List[int]
    inbox: List[InboundMessage]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form — what `read_state()` hands the agent."""
        return {
            "sweep_name": self.sweep_name,
            "workspace": self.workspace,
            "trigger": self.trigger,
            "plan": self.plan,
            "totals": self.totals,
            "trials": self.trials,
            "newly_failed": [asdict(f) for f in self.newly_failed],
            "newly_completed": self.newly_completed,
            "inbox": [asdict(m) for m in self.inbox],
        }


# --- snapshot rotation ------------------------------------------------------

def _hyperherd_dir(workspace: Path) -> Path:
    return workspace / ".hyperherd"


def _rotate_and_capture(workspace: Path) -> Dict[str, Any]:
    """Move the previous snapshot to .prev, fetch a fresh one, write it.

    Returns the parsed current snapshot dict. If no previous snapshot
    exists (first ever tick), the .prev file is left absent — diff
    helpers handle that case.
    """
    hh = _hyperherd_dir(workspace)
    hh.mkdir(parents=True, exist_ok=True)
    cur_path = hh / SNAPSHOT_FILE
    prev_path = hh / PREV_SNAPSHOT_FILE

    if cur_path.exists():
        shutil.copyfile(cur_path, prev_path)

    # Use `python -m hyperherd.cli` rather than the `herd` script so this
    # works regardless of whether `herd` is on PATH (e.g. during dev / tests
    # before `pip install -e .`).
    proc = subprocess.run(
        [sys.executable, "-m", "hyperherd.cli", "snapshot", str(workspace)],
        capture_output=True, text=True, check=True,
    )
    cur_path.write_text(proc.stdout)
    return json.loads(proc.stdout)


def _read_prev(workspace: Path) -> Optional[Dict[str, Any]]:
    prev_path = _hyperherd_dir(workspace) / PREV_SNAPSHOT_FILE
    if not prev_path.is_file():
        return None
    try:
        return json.loads(prev_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --- diff helpers -----------------------------------------------------------

def _indices_with_status(snapshot: Dict[str, Any], status: str) -> set:
    return {t["index"] for t in snapshot.get("trials", []) if t.get("status") == status}


def _diff_failed(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[FailureView]:
    """Indices that newly entered `failed` since the previous snapshot."""
    cur_failed = _indices_with_status(cur, "failed")
    prev_failed = _indices_with_status(prev, "failed") if prev else set()
    new_idx = cur_failed - prev_failed

    cur_trials = {t["index"]: t for t in cur.get("trials", [])}
    failed_stderr = {b["index"]: b for b in cur.get("failed_stderr", [])}

    out: List[FailureView] = []
    for idx in sorted(new_idx):
        trial = cur_trials.get(idx, {})
        block = failed_stderr.get(idx, {})
        out.append(FailureView(
            index=idx,
            experiment_name=trial.get("experiment_name"),
            slurm_state=trial.get("slurm_state"),
            stderr_tail=block.get("stderr_lines") or [],
        ))
    return out


def _diff_completed(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> List[int]:
    cur_done = _indices_with_status(cur, "completed")
    prev_done = _indices_with_status(prev, "completed") if prev else set()
    return sorted(cur_done - prev_done)


# --- plan + inbox -----------------------------------------------------------

def _read_plan(workspace: Path) -> str:
    path = _hyperherd_dir(workspace) / PLAN_FILE
    if not path.is_file():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _drain_inbox(workspace: Path) -> List[InboundMessage]:
    """Read inbox.jsonl, return parsed messages, then truncate the file.

    Truncating rather than deleting keeps the file's inode stable for any
    process that's tailing it for debugging. Lines that fail to parse
    are silently dropped — we'd rather lose one message than abort the
    tick.
    """
    path = _hyperherd_dir(workspace) / INBOX_FILE
    if not path.is_file():
        return []
    try:
        raw = path.read_text()
    except OSError:
        return []

    msgs: List[InboundMessage] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            msgs.append(InboundMessage(
                timestamp=d["timestamp"],
                source=d.get("source", "unknown"),
                author=d.get("author", ""),
                text=d.get("text", ""),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    # Truncate after a successful read.
    try:
        path.write_text("")
    except OSError:
        pass
    return msgs


# --- public entrypoint ------------------------------------------------------

def compute(workspace: Path, trigger: TickTrigger = "scheduled") -> TickState:
    """Build the per-tick state document. One subprocess call to `herd
    snapshot`, one snapshot rotation, plus filesystem reads."""
    workspace = Path(workspace).resolve()
    cur = _rotate_and_capture(workspace)
    prev = _read_prev(workspace)

    return TickState(
        sweep_name=cur.get("sweep_name", "unknown"),
        workspace=str(workspace),
        trigger=trigger,
        plan=_read_plan(workspace),
        totals=cur.get("totals", {}),
        trials=cur.get("trials", []),
        newly_failed=_diff_failed(prev, cur),
        newly_completed=_diff_completed(prev, cur),
        inbox=_drain_inbox(workspace),
    )
