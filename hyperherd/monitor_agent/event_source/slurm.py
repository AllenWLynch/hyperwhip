"""SLURM polling event source.

Periodically asks `herd snapshot` for the sweep state, diffs trial-status
sets against the previous poll, and emits `WakeEvent("failure")` /
`WakeEvent("completion")` when transitions occur.

Why polling instead of `sacct --json` directly: `herd snapshot` already
encapsulates all the manifest-aware logic (slurm job IDs to trial
indices, status mapping, etc.); reusing it keeps this module thin and
ensures the poller and the per-tick state assembler agree on what a
trial's status is.

Coalescing: the daemon drains the queue after each tick (state.compute
already absorbed every transition into newly_failed/newly_completed), so
emitting one event per detected delta is fine — the daemon won't run a
second redundant tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Set

from hyperherd.monitor_agent.event_source import WakeEvent

log = logging.getLogger(__name__)


DEFAULT_POLL_INTERVAL_SECONDS = 60


class SlurmPoll:
    """Poll the workspace for trial state transitions on a fixed interval.

    Stateful — tracks the set of indices it has already seen as terminal,
    so that on startup it doesn't fire spurious events for trials that
    were already failed/completed before the daemon launched.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self._workspace = Path(workspace)
        self._interval = float(interval_seconds)
        self._seen_failed: Set[int] = set()
        self._seen_completed: Set[int] = set()

    async def run(self, queue: asyncio.Queue) -> None:
        """Run forever (or until cancelled). Push WakeEvents into `queue`
        when transitions are detected. Errors are logged and the loop
        continues — a transient sacct hiccup shouldn't kill the daemon."""

        # Initial baseline: capture which trials are *already* terminal so
        # we don't emit a flood of events for pre-existing state.
        try:
            snap = await self._snapshot()
            self._seen_failed = self._failed_set(snap)
            self._seen_completed = self._completed_set(snap)
            log.info(
                "SlurmPoll baseline: %d failed, %d completed",
                len(self._seen_failed), len(self._seen_completed),
            )
        except Exception as e:
            log.warning("SlurmPoll baseline snapshot failed: %s", e)

        while True:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                return

            try:
                snap = await self._snapshot()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("SlurmPoll snapshot error: %s", e)
                continue

            failed = self._failed_set(snap)
            completed = self._completed_set(snap)

            new_failed = failed - self._seen_failed
            new_completed = completed - self._seen_completed

            if new_failed:
                log.info("SlurmPoll: new failed indices %s", sorted(new_failed))
                await queue.put(WakeEvent(trigger="failure"))
            if new_completed:
                log.info("SlurmPoll: new completed indices %s",
                         sorted(new_completed))
                await queue.put(WakeEvent(trigger="completion"))

            self._seen_failed = failed
            self._seen_completed = completed

    async def _snapshot(self) -> dict:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "hyperherd.cli", "snapshot", str(self._workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"`herd snapshot` exit {proc.returncode}: {err.decode().strip()}"
            )
        return json.loads(out)

    @staticmethod
    def _failed_set(snap: dict) -> Set[int]:
        return {
            t["index"] for t in snap.get("trials", [])
            if t.get("status") == "failed"
        }

    @staticmethod
    def _completed_set(snap: dict) -> Set[int]:
        return {
            t["index"] for t in snap.get("trials", [])
            if t.get("status") == "completed"
        }
