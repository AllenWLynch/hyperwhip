# Autonomous monitor

`herd monitor` hands the sweep to a Claude Code agent that operates it for you — start it once, walk away, get pinged on your phone only when something needs you. It's the **recommended way** to run any sweep that takes longer than a few minutes.

## What you get

- **Staged rollout.** The agent submits trial 0 first as a canary, watches it for a few minutes, then submits trials 1–2, then the rest. One bad launcher script doesn't waste your whole array.
- **Failure triage.** When trials fail, the agent groups them by signature and decides what to do: bump `slurm.mem` and resubmit on host OOM, bump `slurm.time` on TIMEOUT, just notify on CUDA OOM (since `slurm.mem` doesn't fix GPU memory), halt the loop on a recurring exception (likely a code bug, not a flake).
- **Live-metric warnings.** NaN/inf in the success metric → the agent kills the trial. Anything else suspicious → it warns you via [`herd msg`](commands.md#herd-msg) and lets you decide. (Proper early stopping is what Hyperband / ASHA / BOHB are for; the agent doesn't pretend to do that.)
- **Phone notifications.** Per-tick status messages plus instant alerts on failure or sweep-done go to whatever webhook is configured (Slack, Discord, ntfy). Zero-config ntfy fallback works out of the box.
- **Always-alive heartbeat.** Even quiet ticks emit a one-line `tick clean — N running, M completed, K failed. Next tick in 30 min.` so a silent channel always means the agent crashed, not "nothing happened."

## Start

```bash
cd ~/sweeps/your_workspace
herd monitor
```

That's it. `herd monitor`:

1. Resolves the ntfy fallback URL (or your configured webhook) and prints it. Paste it into the [ntfy iOS / Android app](https://ntfy.sh) for push notifications.
2. Spawns `herd watch` in the background (pidfile + log file under `.hyperherd/`).
3. Writes agent allow-rules to `<workspace>/.claude/settings.local.json` so unattended ticks don't stall on permission prompts.
4. `exec`s `claude "<prompt>"` — an interactive Claude Code session that's already running the setup interview as its first turn. No paste step.

To survive a logout, wrap the whole thing in tmux:

```bash
tmux new -s monitor 'herd monitor'
# Ctrl-b d to detach. `tmux attach -t monitor` to come back.
```

## Setup interview

On first start, the agent asks five short questions:

1. **Where are live training metrics?** Recommend `wandb`. Also accepted: `.hyperherd/results/<idx>.json` (via [`log_result()`](results.md)), tensorboard event dirs, mlflow tracking URIs, or "I don't have one." Without a live source, only end-of-trial metrics are visible.
2. **Success metric and direction.** e.g. `val_acc maximize`. Used for ranking + NaN/inf detection.
3. **Auto-remediate or notify-only?** When a failure could be fixed by editing `slurm.mem` / `slurm.time`, do you want the agent to bump and resubmit, or just notify and let you decide? **Default: notify.** Many clusters have hard caps that bumping won't beat, and silent YAML edits are unwelcome.
4. **Time budget.** Run until sweep complete, or stop after N hours? Default: until complete.
5. **Notification channel.** Confirm the webhook your `watch:` block is configured for, or accept the auto-generated ntfy fallback.

Answers are persisted to `.hyperherd/MONITOR_PLAN.md` so subsequent ticks (which start with a fresh agent activation) can read them back.

## Per-tick lifecycle

Once rolled out, the agent operates on a self-paced loop. Each tick:

1. Reads `herd snapshot` once (one CLI call → status + sacct + metrics + last-log + recent failed-trial stderr in one JSON document).
2. Diffs against the previous snapshot to find newly-failed and newly-completed trials.
3. Triages failures per the table below; remediates if `Remediation: remediate`.
4. Posts one `herd msg` summarizing the tick, including the next-tick countdown.
5. Calls `ScheduleWakeup(delaySeconds=N)` to schedule the next tick, then ends its turn.

Cadence is **adaptive** — short delays during rollout and right after activity, long delays when the sweep is just running:

| Situation | Next tick |
|---|---|
| Canary just submitted | 2 min |
| Canary still queued / running < 5 min | 3 min |
| Phase 2 ramp | 3 min |
| New failure or completion this tick | 5 min |
| Recent activity within last 2 ticks | 15 min |
| 1 quiet tick | 30 min |
| 3+ consecutive quiet ticks | 60 min (max) |
| Done / halted | (loop ends) |

## Failure triage

| SLURM cause | Cause type | If `remediate` | If `notify` |
|---|---|---|---|
| `OUT_OF_MEMORY` (sacct state) | Host RAM exceeded | Bump `slurm.mem` 50%, resubmit failed indices with `--force` | Notify only — don't touch the YAML |
| `RuntimeError: CUDA out of memory` | GPU VRAM exceeded | **Notify only.** `slurm.mem` controls host RAM, not GPU memory; bumping won't help. User has to reduce batch size or move partitions. | Same |
| `TIMEOUT` | Wall-clock too short | Bump `slurm.time` 50%, resubmit | Notify |
| `NODE_FAIL` / preemption | Infrastructure | Resubmit unchanged | Notify |
| Same Python exception across ≥ 2 trials | Code/env bug | **Halt the loop**, do not resubmit. The user has to fix the trainer. | Same |

Auto-remediation is capped at **one bump per failure class per sweep**. If a 50% mem bump still OOMs, the agent switches that class to notify mode rather than climbing forever.

## Notifications

Two streams converge on the same webhook:

- **`herd watch`** (the background daemon) posts structured event lines instantly: `[<sweep>] trial 3 cancelled (CANCELLED)`, `[<sweep>] sweep complete`, etc. Real-time — sub-poll-interval latency.
- **The agent's `herd msg`** posts every tick (including quiet ones) with a status line plus the next-tick countdown. Adaptive cadence above.

Both prefix every message with the sweep name automatically. The agent additionally prefixes its body with `Herd dog:` so you can tell agent posts from raw watch events at a glance:

```
[mnist_sweep] Herd dog: tick clean — 4 running, 5 completed, 0 failed. Next tick in 30 min.
[mnist_sweep] trial 3 (lr-0.1_opt-sgd) failed (TIMEOUT) — 5 done, 1 failed
[mnist_sweep] Herd dog: bumped slurm.time 1h→1h30m after 1 TIMEOUT (idx 3); resubmitting. Next tick in 5 min.
```

## Stop, restart, override

```bash
herd monitor --stop          # stops the background herd watch (Claude died with its terminal)
herd monitor --no-watch      # start the agent without spawning a new watch (one's already running)
herd monitor --no-auto-allow # don't write the permission allowlist (expect to approve every tool call)
```

To stop the agent loop itself: open the Claude Code session and Ctrl-C, or kill its tmux pane. The watch daemon keeps running unless you `herd monitor --stop`.

To restart from scratch (forget the setup interview, run rollout again): `rm <workspace>/.hyperherd/MONITOR_PLAN.md` then `herd monitor`.

## Troubleshooting

**The session sits silently after I run `herd monitor`.** A tool call hit a permission prompt that didn't get approved. Check the Claude Code session for a pending permission dialog. The agent is told to send a `herd msg` warning *before* any off-list command, so you should also see a "pausing for permission" notification on your phone — accept the prompt or kill the session and the agent will skip that step on the next tick.

**Watch and agent posting to different ntfy topics.** Fixed in current code — `herd monitor` now resolves the topic upfront before spawning anything. If you have a stale split from an earlier version, `herd monitor --stop`, then `rm <workspace>/.hyperherd/watch.json`, then `herd monitor` again.

**`uv.lock` / venv not picked up on workers.** Different problem — see [Activate your environment](launcher.md#activate-your-environment-inside-the-launcher) in the launcher docs. The agent can't fix a launcher that's invoking the wrong Python.

**The agent is "thinking too much" between actions.** It's doing the setup interview or first-tick scan. After `MONITOR_PLAN.md` exists, subsequent ticks are short — read snapshot, decide, post, schedule next, end turn.

## How it works under the hood

- The persistent loop is Claude Code's built-in **dynamic `/loop`** — the agent calls `ScheduleWakeup(delaySeconds=N)` at the end of each tick to schedule its own next wake. There's no external trigger that can poke a parked Claude Code session, so failure-driven wake-ups happen via the *adaptive cadence* (short delays after activity), not via real-time interrupts. `herd watch` covers the real-time alert path.
- All cross-tick state lives in files: `.hyperherd/MONITOR_PLAN.md` (the plan), `.hyperherd/last-snapshot.json` (last tick's state for delta detection), `.hyperherd/last-snapshot.prev.json` (the one before that). The agent has no in-memory continuity between ticks.
- The agent's playbook lives in the [`hyperherd-monitor` skill](claude-skill.md#what-hyperherd-monitor-does). Read `~/.claude/skills/hyperherd-monitor/SKILL.md` (or `hyperherd/skills/hyperherd-monitor/SKILL.md` in the repo) to see exactly what it's told.
