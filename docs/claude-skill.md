# Claude Code skills

HyperHerd ships **two** [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills):

- **`hyperherd-config`** — teaches Claude how to author and edit `hyperherd.yaml` files. Open a Claude Code session and ask "set up a sweep over learning rate and three optimizers" — Claude uses the skill's checklist + patterns to produce a config that follows HyperHerd conventions.
- **`hyperherd-monitor`** — drives the autonomous sweep operator invoked by [`herd monitor`](commands.md#herd-monitor). The skill defines the staged rollout, failure-triage table, NaN/inf detection, status-report cadence, and the exact bash allowlist the agent stays within. You don't usually invoke it by hand — `herd monitor` exec's `claude` with the right initial prompt for you.

## Install

```bash
herd install-skill
```

That installs both skills into `~/.claude/skills/hyperherd-{config,monitor}/`. Open a new Claude Code session anywhere and they're available.

| Flag | Description |
|------|-------------|
| `--scope user` *(default)* | Writes to `~/.claude/skills/<skill>/`. Available in every Claude Code session. |
| `--scope project` | Writes to `./.claude/skills/<skill>/` in the current directory. Commit it and collaborators in this repo get the skill. |
| `--name <skill>` | Install only the named skill (`hyperherd-config` or `hyperherd-monitor`). Default installs both. |
| `-f, --force` | Overwrite an existing install. Use this after upgrading HyperHerd to pick up skill updates. |

### Re-installing after a HyperHerd upgrade

Both skills are shipped as package data inside the `hyperherd` Python package. After upgrading HyperHerd, run `herd install-skill --force` to pick up any updates.

## What `hyperherd-config` does

The config skill is a structured prompt that:

1. Walks Claude through the **sweep shape** decision (full grid / partial grid / one-at-a-time) and the implications for `default` fields.
2. Documents the parameter schema (discrete vs continuous, `abbrev`, `labels`, log scale).
3. Provides patterns for the four condition forms (`exclude`, `force`, `set`, `when` matchers).
4. Gives launcher templates for common environments.
5. Points Claude at the [Sweep config reference](configuration.md) for any field it's unsure about.

It is intentionally a **checklist + patterns**, not a substitute for the docs. For non-trivial configs Claude is instructed to read the full reference first.

## What `hyperherd-monitor` does

The monitor skill is the playbook the agent follows on every wake-up:

- **Setup interview** on first invocation (metric source, success metric, remediate-vs-notify, time budget, channel) — persisted to `.hyperherd/MONITOR_PLAN.md` for subsequent ticks.
- **Phased rollout** as a state machine across ticks: canary `-i 0` → small batch `-i 1-2` → full sweep, advancing one phase per tick.
- **Failure triage table** with separate handling for host OOM (bumpable), CUDA OOM (notify-only — `slurm.mem` doesn't fix GPU memory), `TIMEOUT`, `NODE_FAIL`, and recurring exception clusters.
- **NaN/inf-only kill policy** for live trials. Anything else suspicious is a `herd msg` warning, not an action — proper early stopping is what algorithms like Hyperband / ASHA / BOHB are for.
- **Adaptive cadence.** Tight delays during rollout (3–5 min), backed off to ~30 min during steady-state, 60 min after several quiet ticks.
- **Approved-tooling pre-flight.** Before any non-`herd` Bash call, the agent checks the proposed command against a baked-in allowlist; if it's off-list, it warns the user via `herd msg` first so an unattended tick never silently stalls on a permission prompt.

The full skill source is the canonical reference. After install, read `~/.claude/skills/hyperherd-monitor/SKILL.md` directly to see exactly what the agent is told.

## Source

The canonical skill sources live in the HyperHerd repo at `hyperherd/skills/<name>/SKILL.md`. The repo's own `.claude/skills/<name>/SKILL.md` files are symlinks to them, so working in a checkout of the HyperHerd repo Just Works without running `install-skill`.
