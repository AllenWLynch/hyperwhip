# HyperHerd

**Hyperparameter sweeps on SLURM, run by an autonomous agent.** Declare your search in YAML, hand over a one-line launcher script, and walk away — `herd monitor` submits trials in stages, diagnoses failures, retries the ones SLURM can fix, and pings you on Discord only when it can't.

📖 **Full documentation: [allenwlynch.github.io/hyperherd](https://allenwlynch.github.io/hyperherd/)**

## What you get

- **One-command sweeps.** Write a YAML, run `herd run`, and that's it — no sbatch boilerplate, no manual resubmits.
- **An agent that operates the sweep for you.** `herd monitor` ramps trials in stages, diagnoses failures, bumps memory or wall-time when that's the right fix, and only interrupts you when it isn't.
- **Two-way Discord control.** A dedicated channel per sweep with deterministic slash commands (`/status`, `/run`, `/cancel`, `/tail`, …) and free-form mentions for the agent.
- **Resume from anywhere.** Pull the plug, edit the sweep, re-run — completed trials stick, failed ones go back to the queue.
- **Edit mid-run.** Bump a learning-rate range or add a value; the next `herd run` appends the new trials without disturbing the ones already running.
- **Configs you don't have to memorize.** A bundled Claude Code skill writes `hyperherd.yaml` for you from a one-paragraph description.

[Hydra](https://hydra.cc/) is the recommended trainer harness (its CLI consumes the override format natively), but the launcher is free-form bash — parse the arguments however you want.

## Quick start

```bash
# Install (Python 3.8+ for the base CLI)
pip install hyperherd

# Install the Claude Code skill for authoring sweep configs
herd install-skill

# Scaffold a workspace
herd init my_experiment

# Edit my_experiment/hyperherd.yaml and my_experiment/launch.sh, then:
herd run my_experiment --dry-run    # preview
herd run my_experiment              # submit
herd status my_experiment           # one-shot status
```

To run the autonomous monitor (Python ≥ 3.10 + a Discord bot — see [Discord setup](https://allenwlynch.github.io/hyperherd/discord-setup/)):

```bash
pip install 'hyperherd[monitor]'
herd monitor my_experiment
```

## Documentation

- [Getting started](https://allenwlynch.github.io/hyperherd/getting-started/) — install, scaffold, run your first sweep
- [Autonomous monitor](https://allenwlynch.github.io/hyperherd/monitor/) — the agent daemon: setup, Discord control, failure triage
- [Discord setup](https://allenwlynch.github.io/hyperherd/discord-setup/) — one-time bot creation walkthrough
- [Sweep config reference](https://allenwlynch.github.io/hyperherd/configuration/) — every field in `hyperherd.yaml`
- [Conditions](https://allenwlynch.github.io/hyperherd/conditions/) — filter or modify parameter combinations
- [Launcher script](https://allenwlynch.github.io/hyperherd/launcher/) — contract + examples (Apptainer, conda, non-Hydra)
- [Command reference](https://allenwlynch.github.io/hyperherd/commands/) — every `herd` subcommand
- [Claude Code skill](https://allenwlynch.github.io/hyperherd/claude-skill/) — authoring configs by asking Claude

## Requirements

- Python ≥ 3.8 for the base `herd` CLI
- Python ≥ 3.10 for the `[monitor]` extras (the autonomous monitor — Discord, Claude Agent SDK)
- A SLURM cluster with `sbatch`, `sacct`, `squeue`, `scancel` on the submission host

## Releasing

Versions are derived from git tags via `setuptools_scm` — there's nothing to bump in `pyproject.toml`. To cut a release:

```bash
git tag -a v0.1.0 -m "first release"
git push origin v0.1.0
```

The `Publish to PyPI` GitHub Action picks up the tag, builds an sdist + wheel with version `0.1.0`, verifies the built version matches the tag, and uploads to PyPI via Trusted Publisher (OIDC — no API token in repo secrets).

One-time PyPI setup before the first release:
1. Create the `hyperherd` project on [PyPI](https://pypi.org).
2. In the project's "Publishing" settings, add a Trusted Publisher: owner `AllenWLynch`, repo `hyperherd`, workflow `publish.yml`, environment `pypi`.
