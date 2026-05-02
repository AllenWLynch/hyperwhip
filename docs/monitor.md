# Autonomous monitor

`herd monitor` hands the sweep to a Claude Code agent that operates it for you. Start it once, walk away, get pinged on your phone only when something needs you. **Recommended for any sweep longer than a few minutes.**

## What it does

- **Staged rollout.** Trial 0 first as a canary, then trials 1–2, then the rest. One bad launcher script doesn't waste the whole array.
- **Failure triage.** On host OOM the agent bumps `slurm.mem` and resubmits. On TIMEOUT it bumps `slurm.time`. CUDA OOM is notified, not retried (`slurm.mem` doesn't fix GPU memory). Recurring identical exceptions halt the loop — likely a code bug worth your attention.
- **NaN/inf-only kill policy** for live trials. Anything else suspicious gets a warning to your phone, not an action.
- **Phone notifications** via Slack / Discord / ntfy on every tick (so silence means crash, not "nothing happened") and instantly on every failure.

## Start

```bash
cd ~/sweeps/your_workspace
herd monitor
```

Wrap in `tmux` to survive a logout:

```bash
tmux new -s monitor 'herd monitor'
# Ctrl-b d to detach. `tmux attach -t monitor` to come back.
```

The first invocation prints your notification URL. Paste it into the [ntfy iOS / Android app](https://ntfy.sh) for push notifications.

## Setup interview

On first start the agent asks five short questions, in one message:

1. **Where are live training metrics?** Recommend `wandb`. Also accepted: `.hyperherd/results/<idx>.json` (via [`log_result()`](results.md)), tensorboard, mlflow, or "I don't have one."
2. **Success metric and direction.** e.g. `val_acc maximize`.
3. **Auto-remediate or notify-only?** Should the agent edit `slurm.mem` / `slurm.time` and resubmit on fixable failures, or just notify and let you decide? **Default: notify.**
4. **Time budget.** Run until the sweep finishes, or stop after N hours.
5. **Notification channel.** Confirm the configured webhook or accept the auto-generated ntfy fallback.

## What you'll see on your phone

Every tick posts a status line, even quiet ones — silence means the agent crashed:

```
[mnist_sweep] Herd dog: tick clean — 4 running, 5 completed, 0 failed. Next tick in 30 min.
[mnist_sweep] trial 3 (lr-0.1_opt-sgd) failed (TIMEOUT) — 5 done, 1 failed
[mnist_sweep] Herd dog: bumped slurm.time 1h→1h30m after 1 TIMEOUT (idx 3); resubmitting. Next tick in 5 min.
```

Tick cadence is adaptive: short (3–5 min) during rollout and just after activity, long (30–60 min) when the sweep is running steady-state. The "Next tick in …" countdown is in every message so you know whether to expect something soon or go to lunch.

## Failure triage

| SLURM cause | Cause type | If `remediate` | If `notify` |
|---|---|---|---|
| `OUT_OF_MEMORY` | Host RAM exceeded | Bump `slurm.mem` 50%, resubmit | Notify only |
| `RuntimeError: CUDA out of memory` | GPU VRAM | **Notify only**, even in `remediate` mode | Notify |
| `TIMEOUT` | Wall-clock too short | Bump `slurm.time` 50%, resubmit | Notify |
| `NODE_FAIL` / preemption | Infrastructure | Resubmit unchanged | Notify |
| Same exception ≥ 2 trials | Code/env bug | **Halt the loop** — fix the trainer | Same |

Auto-bumps are capped at one per failure class per sweep.

## Stop, restart, override

```bash
herd monitor --stop          # stop the background watch
herd monitor --no-watch      # skip starting watch (one's already running)
herd monitor --no-auto-allow # require interactive permission approval
```

Stop the agent itself by killing the Claude Code session (Ctrl-C, or close the tmux pane).

To run the setup interview again from scratch: `rm <workspace>/.hyperherd/MONITOR_PLAN.md` then `herd monitor`.

## Troubleshooting

**Channel is silent.** A tool call hit a permission prompt that hasn't been accepted. The agent posts a "pausing for permission" warning before any off-list command — accept the prompt in the Claude Code session, or kill the session and the agent will skip that step on the next tick.

**Trial fails with `ModuleNotFoundError`.** The launcher isn't activating your venv on the worker. See [Activate your environment](launcher.md#activate-your-environment-inside-the-launcher).

**The agent's playbook lives in the [`hyperherd-monitor` skill](claude-skill.md#what-hyperherd-monitor-does).** If you want to tweak its behavior — different cadence, different triage policy, custom warnings — edit `~/.claude/skills/hyperherd-monitor/SKILL.md`.
