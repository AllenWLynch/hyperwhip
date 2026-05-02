# HyperHerd

**Hyperparameter sweeps on SLURM, run by an autonomous agent.** Declare your search in YAML, hand over a one-line launcher script, and walk away — [`herd monitor`](commands.md#herd-monitor) spawns a Claude Code agent that submits trials in stages, diagnoses failures, retries when SLURM can fix the problem, and pings your phone only when it can't.

## What you write

Two files in a workspace directory:

- **`hyperherd.yaml`** — your sweep declaratively: parameters, grid mode, SLURM resources, conditions.
- **`launch.sh`** — a one-line bash script that receives a `name=value` override string as `$1` and runs your training command in whatever environment you need (container, conda, uv, modules).

## What you get

- **One-command sweeps.** No sbatch boilerplate, no manual resubmits — `herd run` generates and submits the array, tracks state, and resumes failed/pending trials on rerun.
- **An agent that actually operates the sweep.** [`herd monitor`](commands.md#herd-monitor) ramps trials in stages, diagnoses failures, bumps memory or wall-time when that's the right fix, and only interrupts you when it isn't.
- **Phone notifications.** Real-time Slack / Discord / ntfy alerts on failure and completion via [`herd watch`](commands.md#herd-watch). Zero-config ntfy fallback works out of the box.
- **Edit your sweep mid-run.** Bump a parameter range or add a value; the next `herd run` appends new trials without touching the ones already running.
- **Configs you don't have to memorize.** The bundled [Claude Code skill](claude-skill.md) writes `hyperherd.yaml` for you from a one-paragraph description.
- **An audit trail.** Every trial's parameters, status, and logged metrics live in `.hyperherd/` and come out as TSV via `herd res` or JSON via `herd snapshot`.

[Hydra](https://hydra.cc/) is the recommended trainer harness — its CLI consumes `name=value` overrides natively, so the string passes through unchanged — but the launcher is free-form bash, so parse the arguments however you want.

## Scope

HyperHerd is opinionated. It assumes:

1. SLURM job arrays as the dispatch mechanism.
2. `name=value` overrides as the parameter contract.
3. A bash launcher script as the integration point.

## Where to next

<div class="grid cards" markdown>

- **[Autonomous monitor](monitor.md)** — start here if you want the agent runner. Lifecycle, failure-triage policy, notifications.
- **[Getting started](getting-started.md)** — install, scaffold, run your first sweep.
- **[Sweep config reference](configuration.md)** — every field in `hyperherd.yaml`.
- **[Command reference](commands.md)** — every `herd` subcommand.
- **[Claude Code skills](claude-skill.md)** — generate configs by asking Claude; drive the monitor agent.

</div>
