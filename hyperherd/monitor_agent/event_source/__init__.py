"""Event sources that wake the daemon between scheduled ticks.

Each source runs as a coroutine and pushes `WakeEvent`s into the daemon's
queue. The daemon races the queue against its scheduled-tick timeout —
whichever fires first determines the next tick's `trigger` field, so the
agent sees why it was woken.

Sources currently shipping:

- `slurm.SlurmPoll` — polls `herd snapshot` for trial-state transitions
  into `failed` / `completed` and emits the corresponding event.
- The Discord channel's inbox writer pushes `WakeEvent("user_message")`
  via the `on_write` callback in `channel.make_inbox_writer`.

Phase 5 may add wandb-driven anomaly events (NaN/inf detection on a live
metric stream), routed through the same queue.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WakeEvent:
    """One wake-up reason. The `trigger` is what the daemon hands to
    `state.compute` so the agent can see why it's running this tick."""

    trigger: str  # "scheduled" / "failure" / "completion" / "user_message" / "boot"
