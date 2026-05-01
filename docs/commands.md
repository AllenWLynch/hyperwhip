# Command reference

Every `herd` subcommand is documented below. Most subcommands take an optional `workspace` directory as their first positional argument (default: `.`); when run from inside the workspace you can usually omit it.

## `herd init`

Scaffold a new sweep workspace.

```bash
herd init [DIRECTORY] [--config FILE] [--launcher FILE] [-f]
```

Creates `DIRECTORY/hyperherd.yaml` and `DIRECTORY/launch.sh` with template content; the experiment name is taken from the directory name. If `DIRECTORY` is omitted, files are written to the current directory.

The templates have placeholder SLURM resource fields (`partition`, `time`, `mem`, `cpus_per_task`) you'll edit to match your cluster — `herd init` doesn't try to be a substitute for opening the YAML.

| Flag | Description |
|------|-------------|
| `--config FILE` | Copy `FILE` in as `hyperherd.yaml` instead of generating a template — useful for cloning an existing sweep |
| `--launcher FILE` | Copy `FILE` in as `launch.sh` instead of generating a template |
| `-f, --force` | Overwrite existing files in the target directory |

## `herd run`

Submit (or resubmit) the sweep.

```bash
herd run [WORKSPACE] [flags]
```

Generates the trial manifest, runs preflight checks, writes the sbatch script to `.hyperherd/job.sbatch`, submits it, and records the SLURM job ID.

`herd run` is idempotent: it only submits trials whose status is `ready`, `failed`, or `cancelled`. Trials that are `submitted`, `queued`, `running`, or `completed` are skipped unless you opt in with `--force`.

| Flag | Description |
|------|-------------|
| `-n, --dry-run` | Print the sbatch script and trial list; don't submit |
| `-j, --max-concurrent N` | Cap concurrent running tasks (overrides `slurm.max_concurrent`) |
| `-i, --indices SPEC` | Submit only these trial indices, e.g. `0-3,5,7-9` |
| `-f, --force` | With `--indices`, allow resubmitting running/completed trials. Without, allow config edits that drop running/completed trials (kept as orphans). |

**Editing the config mid-sweep is supported.** If you edit `hyperherd.yaml` between runs, `herd run` reconciles the new manifest against the old one: new trials are appended, removed trials are dropped (or kept as orphans with `-f` if they were already running/completed). See [Re-running and reconciliation](workspace.md#re-running-and-reconciliation) for the rules.

## `herd status`

Show the current status table for every trial.

```bash
herd status [WORKSPACE]
```

Status values:

| Status | Meaning |
|--------|---------|
| `ready` | Never submitted |
| `submitted` | Sent to SLURM, not yet picked up |
| `queued` | SLURM `PENDING` |
| `running` | SLURM `RUNNING` |
| `completed` | SLURM `COMPLETED` |
| `failed` | SLURM `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY` / `NODE_FAIL` |
| `cancelled` | SLURM `CANCELLED` (or via `herd stop`) |

`herd status` syncs from SLURM each time it runs (via `sacct`).

## `herd stats`

Print runtime + memory accounting for one or all trials, sourced from `sacct`.

```bash
herd stats [WORKSPACE] [INDEX]
```

Columns: index, state, elapsed, max RSS (GB), avg RSS (GB), requested mem (GB), experiment name. Memory values are converted from sacct's raw units to gigabytes.

## `herd tail`

Print the last N lines of a trial's stdout log.

```bash
herd tail [WORKSPACE] INDEX [-n LINES]
```

Reads from `.hyperherd/logs/<index>.out`. Use `-n` (default 20) to control how many lines.

## `herd res`

Print a TSV of every trial's parameters and logged metrics.

```bash
herd res [WORKSPACE]
```

Combines `manifest.json` (parameters, experiment name) with `.hyperherd/results/*.json` (metrics written by [`log_result()`](results.md)). Trials without results show empty cells.

## `herd test`

Run a single trial locally (no SLURM) via the configured launcher.

```bash
herd test [WORKSPACE] [INDEX] [--cfg-job]
```

Default `INDEX` is 0. The launcher is invoked exactly as the SLURM array would invoke it, so this is the right place to debug the launcher script itself, exercise the trainer end-to-end on a login node, or verify a fix before resubmitting the array.

For safety, `herd test` refuses any index that has previously been submitted to SLURM — running again would clobber its outputs and logs. Pick a different index, or `herd clean --all` first.

| Flag | Description |
|------|-------------|
| `--cfg-job` | Append `--cfg job` to the override string. For Hydra trainers, this prints the fully resolved config and exits without running training — handy for catching unknown parameter names, type mismatches, or missing required fields. Because nothing real runs, the previously-submitted guard is skipped in this mode. **Hydra-specific** — has no effect on launchers whose trainers don't recognize `--cfg job`. |

`herd test` runs on the login node, so your launcher's environment must be accessible there. If your launcher requires a GPU container that isn't available on the login node, adapt it to gate the heavy parts on a `HYPERHERD_TEST` flag, or test manually.

## `herd stop`

Cancel a running/queued trial.

```bash
herd stop [WORKSPACE] INDEX
herd stop [WORKSPACE] --all
```

Calls `scancel <jobid>_<index>` and updates the manifest to `cancelled`. Pass either an `INDEX` or `--all`, not both. With `--all`, every trial whose status is in (`submitted`, `queued`, `running`) is cancelled.

## `herd clean`

Cancel jobs and clean up workspace state.

```bash
herd clean [WORKSPACE] [-l] [-a]
```

| Flag | Description |
|------|-------------|
| *(none)* | Cancel any running jobs but leave the manifest in place |
| `-l, --logs` | Also remove `.hyperherd/logs/` |
| `-a, --all` | Remove the entire `.hyperherd/` state directory |

`herd clean -a` is destructive — manifests, results, and logs are gone after.

## `herd install-skill`

Install the [Claude Code skill](claude-skill.md) for authoring sweep configs.

```bash
herd install-skill [--scope user|project] [-f]
```

Default scope is `user` (writes to `~/.claude/skills/hyperherd-config/SKILL.md`); `project` writes to `./.claude/skills/`. Use `-f` to overwrite an existing install.
