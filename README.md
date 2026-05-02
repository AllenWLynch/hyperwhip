# HyperHerd

**Hyperparameter sweeps on SLURM, run by an autonomous agent.** Declare your search in YAML, hand over a one-line launcher script, and walk away — `herd monitor` spawns a Claude Code agent that submits trials in stages, diagnoses failures, retries the ones SLURM can fix, and pings your phone only when it can't.

📖 **Full documentation: [allenwlynch.github.io/hyperherd](https://allenwlynch.github.io/hyperherd/)**

## What you get

- **One-command sweeps.** Write a YAML, run `herd run`, and that's it — no sbatch boilerplate, no manual resubmits.
- **An agent that operates the sweep for you.** `herd monitor` ramps trials in stages, diagnoses failures, bumps memory or wall-time when that's the right fix, and only interrupts you when it isn't.
- **Phone notifications.** Real-time Slack / Discord / ntfy alerts on failure and completion. Zero-config ntfy fallback works out of the box; the iOS / Android app pings you the moment something breaks.
- **Resume from anywhere.** Pull the plug, edit the sweep, re-run — completed trials stick, failed ones go back to the queue.
- **Edit mid-run.** Bump a learning-rate range or add a value; the next `herd run` appends the new trials without disturbing the ones already running.
- **Configs you don't have to memorize.** A bundled Claude Code skill writes `hyperherd.yaml` for you from a one-paragraph description.

[Hydra](https://hydra.cc/) is the recommended trainer harness (its CLI consumes the override format natively), but the launcher is free-form bash — parse the arguments however you want.

## Quick start

```bash
# Install
pip install git+https://github.com/AllenWLynch/hyperherd.git

# Install the Claude Code skill for authoring sweep configs
herd install-skill

# Scaffold a workspace
herd init my_experiment

# Edit my_experiment/hyperherd.yaml and my_experiment/launch.sh, then:
herd run my_experiment --dry-run    # preview
herd run my_experiment              # submit
herd status my_experiment           # monitor
```

## Documentation

- [Getting started](https://allenwlynch.github.io/hyperherd/getting-started/) — install, scaffold, run your first sweep
- [Sweep config reference](https://allenwlynch.github.io/hyperherd/configuration/) — every field in `hyperherd.yaml`
- [Conditions](https://allenwlynch.github.io/hyperherd/conditions/) — filter or modify parameter combinations
- [Launcher script](https://allenwlynch.github.io/hyperherd/launcher/) — contract + examples (Apptainer, conda, non-Hydra)
- [Command reference](https://allenwlynch.github.io/hyperherd/commands/) — every `herd` subcommand
- [Claude Code skill](https://allenwlynch.github.io/hyperherd/claude-skill/) — authoring configs by asking Claude

## Requirements

- Python ≥ 3.8
- SLURM cluster with `sbatch`, `sacct`, `squeue`, `scancel` on the submission host
