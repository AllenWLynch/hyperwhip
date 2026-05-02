# Command reference

Every `herd` subcommand is documented below. Most subcommands take an optional `workspace` directory as their first positional argument (default: `.`); when run from inside the workspace you can usually omit it.

## Agent (`--json`) mode

The read- and run-style commands (`run`, `status`, `stats`, `tail`, `res`, `stop`) accept a `--json` flag that emits a structured JSON document to stdout instead of the human-formatted table. Use it when an agent or other automation is driving HyperHerd:

- Numeric memory in **bytes** (not `1.50G`); elapsed time in **seconds** (not `01:30:00`).
- Status uses the stable internal enum: `ready`, `submitted`, `queued`, `running`, `completed`, `failed`, `cancelled`.
- Empty / unknown values come through as `null`, never an empty string.
- Errors still go to stderr with a non-zero exit code; stdout in JSON mode is always a single valid JSON document or empty.
- Warnings (preflight, partition checks) print to stderr as in normal mode and **do not** corrupt the stdout JSON, so `herd run --dry-run --json | jq ...` is always safe.

The JSON shape for each command is documented inline below.

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

**Example output**

--8<-- "_outputs/init.html"

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

**Example output** — successful submission

--8<-- "_outputs/launch.html"

**Example output** — `--dry-run`

--8<-- "_outputs/dry-run.html"

**Agent mode** — `herd run --dry-run --json` emits the full enumeration of hparam combinations the sweep would submit, without touching SLURM. The intended workflow for an agent is to call this, inspect the candidate trials, then call `herd run --json` to actually submit.

```json
{
  "dry_run": true,
  "slurm_job_id": null,
  "sbatch_path": null,
  "submitted_indices": [0, 1, 2, 3],
  "sbatch_script": "#!/bin/bash\n#SBATCH --array=0-3\n...",
  "trials": [
    {"index": 0, "status": "ready", "experiment_name": "lr-0.01_bs-32",
     "params": {"lr": 0.01, "bs": 32}},
    {"index": 1, "status": "ready", "experiment_name": "lr-0.01_bs-64",
     "params": {"lr": 0.01, "bs": 64}}
  ]
}
```

A real (non-dry-run) `herd run --json` returns the same shape with `dry_run: false`, `slurm_job_id` populated, `sbatch_path` set to where the script was written (`.hyperherd/job.sbatch`), and `sbatch_script: null`.

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

**Example output**

--8<-- "_outputs/status.html"

**Agent mode** — `herd status --json`:

```json
{
  "totals": {"total": 11, "running": 4, "completed": 5, "failed": 1, "queued": 1},
  "trials": [
    {"index": 0, "status": "completed", "experiment_name": "lr-0.001_opt-adam",
     "params": {"lr": 0.001, "optimizer": "adam"},
     "last_log_line": "Test acc: 0.978"}
  ]
}
```

## `herd stats`

Print runtime + memory accounting for one or all trials, sourced from `sacct`.

```bash
herd stats [WORKSPACE] [INDEX]
```

Columns: index, state, elapsed, max RSS (GB), avg RSS (GB), requested mem (GB), experiment name. Memory values are converted from sacct's raw units to gigabytes.

**Example output**

--8<-- "_outputs/stats.html"

**Agent mode** — `herd stats --json` emits memory in bytes and elapsed time in seconds, with the SLURM state and the original sacct strings preserved so callers don't have to re-derive them:

```json
{
  "trials": [
    {"index": 0, "experiment_name": "lr-0.001_opt-adam",
     "slurm_state": "COMPLETED",
     "elapsed": "00:01:30", "elapsed_seconds": 90,
     "max_rss_bytes": 1610612736, "ave_rss_bytes": 858993459,
     "req_mem_bytes": 1610612736}
  ]
}
```

## `herd tail`

Print the last N lines of a trial's logs.

```bash
herd tail [WORKSPACE] INDEX [-n LINES] [--stdout | --stderr]
```

By default `herd tail` prints both `.hyperherd/logs/<index>.out` (stdout) and `.err` (stderr), each prefixed by a labelled header. Use `--stdout` or `--stderr` (mutually exclusive) to restrict to one stream. `-n` (default 20) controls how many lines per stream.

**Agent mode** — `herd tail --json` returns each requested stream's path and lines as a structured payload. A stream that doesn't exist on disk shows up with `lines: null` so an agent can distinguish "no log file" from "empty log file":

```json
{
  "index": 3,
  "status": "failed",
  "experiment_name": "lr-0.1_opt-sgd",
  "streams": {
    "stdout": {"path": ".hyperherd/logs/3.out", "lines": ["epoch 1", "..."], "requested": 20},
    "stderr": {"path": ".hyperherd/logs/3.err", "lines": ["RuntimeError: CUDA OOM"], "requested": 20}
  }
}
```

## `herd res`

Print a TSV of every trial's parameters and logged metrics.

```bash
herd res [WORKSPACE]
```

Combines `manifest.json` (parameters, experiment name) with `.hyperherd/results/*.json` (metrics written by [`log_result()`](results.md)). Trials without results show empty cells.

**Agent mode** — `herd res --json` emits one entry per trial (including those without logged metrics, with `metrics: {}`):

```json
{
  "trials": [
    {"index": 0, "experiment_name": "lr-0.001_opt-adam",
     "params": {"lr": 0.001, "optimizer": "adam"},
     "metrics": {"test_acc": 0.978, "test_loss": 0.071}}
  ]
}
```

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

**Agent mode** — `herd stop --json` returns one record per cancelled trial (empty list if there was nothing live):

```json
{
  "cancelled": [
    {"index": 3, "slurm_job_id": "12345", "previous_status": "running"},
    {"index": 7, "slurm_job_id": "12345", "previous_status": "queued"}
  ]
}
```

## `herd watch`

Poll the manifest and post trial state changes to a webhook (Slack, Discord, ntfy.sh, or any URL that accepts a POST).

```bash
herd watch [WORKSPACE] [--once] [--pidfile PATH]
```

All settings come from the `watch:` block in `hyperherd.yaml` — webhook URL, payload format, poll interval, heartbeat cadence, which events to deliver, and the optional Claude-summary flag. See [watch](configuration.md#watch) for the full field reference.

`herd watch` runs in the **foreground** and logs one event line per tick to stdout (`[2026-05-01T14:23:10Z] sweep_done — ...`), so a redirected log gives you a clean event journal.

| Flag | Description |
|------|-------------|
| `--once` | Run a single poll and exit. Use this with `cron`/`systemd timer` if you'd rather schedule polling externally. |
| `--pidfile PATH` | Write the daemon PID to `PATH` for external supervisors / kill scripts. |

### Running persistently on a login node

The cluster-friendly recipe is `nohup` + `--pidfile`: the daemon survives logout, the redirected log doubles as the event journal, and the pidfile gives you a single place to look the process up later.

```bash
cd ~/sweeps/mnist_sweep

nohup herd watch \
    --pidfile .hyperherd/watch.pid \
    > .hyperherd/watch.log 2>&1 &

disown   # detach from this shell so logout won't HUP it
```

Check that it's still running:

```bash
ps -p "$(cat .hyperherd/watch.pid)"
tail -f .hyperherd/watch.log
```

Stop it cleanly:

```bash
kill "$(cat .hyperherd/watch.pid)"
```

The daemon removes its own pidfile on shutdown. If a stale pidfile is left behind (e.g. after `kill -9`) and `ps` says the PID isn't running, just `rm .hyperherd/watch.pid`.

Prefer an interactive session you can reattach to? `tmux new -d -s watch 'herd watch'` and `tmux attach -t watch` work too — but you lose the pidfile + redirected log, so you have to remember the session name.

### Zero-config setup

If `watch.webhook` is unset (or the entire `watch:` block is missing), `herd watch` generates a random `https://ntfy.sh/herd-{slug}-{random}` topic on first run, persists it in the workspace, and prints the URL with subscribe instructions:

```
watch.webhook is unset — falling back to a per-workspace ntfy.sh topic.

    https://ntfy.sh/herd-mnist_sweep-7Hf3kPzQ8w

Subscribe by either:
  • Open the URL in a browser
  • iOS/Android: install the ntfy app and add the topic
  • curl -s https://ntfy.sh/herd-mnist_sweep-7Hf3kPzQ8w/json
```

The same URL is reused across daemon restarts. Anyone with the URL can read the notifications, so use a private webhook for sensitive sweeps.

### Defaults at a glance

With no `watch:` block in your config, you get:

- 60 s poll interval
- Per-trial alert on `failed` / `cancelled`, annotated with the SLURM cause (`TIMEOUT`, `OUT_OF_MEMORY`, `SIGSEGV`, `exit code 1`, ...) and the tail of the trial's stderr
- A `done` notification when the sweep finishes
- A heartbeat digest every 5 minutes (suppressed if nothing changed)
- Posted to the auto-generated ntfy.sh topic

Override any of those by adding a `watch:` block — see the [configuration reference](configuration.md#watch). Failure payload shape and the optional Claude diagnosis are documented under [Failure diagnosis](configuration.md#failure-diagnosis).

## `herd snapshot`

Bundle every read-style command's output (status + sacct + logged metrics + per-trial last-log line + recent failed-trial stderr) into a single JSON document.

```bash
herd snapshot [WORKSPACE] [-n LINES] [--max-failed N]
```

`herd snapshot` is **JSON-only**: it has no human-formatted form. It exists for agent loops where one cheap CLI call per tick beats firing four (`status`, `stats`, `res`, `tail`) and re-stitching the results — and avoids partial-state races between calls when SLURM transitions a trial mid-snapshot.

| Flag | Description |
|------|-------------|
| `-n, --lines` | Max stderr lines to include per failed trial (default: 20) |
| `--max-failed` | Cap on number of failed trials to attach stderr for (default: 20) |

Shape:

```json
{
  "sweep_name": "mnist_sweep",
  "workspace": "/home/you/sweeps/mnist_sweep",
  "totals": {"total": 11, "running": 4, "completed": 5, "failed": 2},
  "trials": [
    {
      "index": 0, "status": "completed", "experiment_name": "lr-0.001_opt-adam",
      "params": {"lr": 0.001, "optimizer": "adam"},
      "slurm_job_id": "12345",
      "slurm_state": "COMPLETED",
      "elapsed": "00:01:30", "elapsed_seconds": 90,
      "max_rss_bytes": 1610612736, "ave_rss_bytes": 858993459,
      "req_mem_bytes": 1610612736,
      "metrics": {"test_acc": 0.978, "test_loss": 0.071},
      "last_log_line": "Test acc: 0.978"
    }
  ],
  "failed_stderr": [
    {
      "index": 5,
      "stderr_path": ".hyperherd/logs/5.err",
      "stderr_lines": ["RuntimeError: CUDA out of memory", "..."],
      "stderr_truncated": false
    }
  ]
}
```

`metrics` is whatever the trial called `log_result()` with — empty dict for trials that haven't logged anything yet (not silently dropped). `last_log_line` is the same one-liner the human `herd status` table shows in its rightmost column. `failed_stderr` is keyed by index in ascending order; an agent that wants to group failures by root cause should fingerprint these stderr blocks.

## `herd monitor`

Start the agent-driven sweep monitor: spawn `herd watch` in the background, print the prompt to paste into Claude Code, and `exec` `claude` so the user lands in an interactive session with the [`hyperherd-monitor` skill](claude-skill.md) ready to run.

```bash
herd monitor [WORKSPACE] [--no-watch]
herd monitor --stop
```

The agent runs under **dynamic `/loop`** (no fixed cadence) and self-paces via `ScheduleWakeup` — short delays during canary rollout and right after activity (~3–5 min), longer delays once the sweep is steady-state running (~30–60 min). The cadence rules live in the `hyperherd-monitor` skill.

| Flag | Description |
|------|-------------|
| `--no-watch` | Skip the background `herd watch` spawn (use if one is already running) |
| `--no-auto-allow` | Don't write monitor allow-rules to `<workspace>/.claude/settings.local.json` (without these rules, Claude Code prompts before each tool call — defeats unattended operation) |
| `--stop` | Kill the background `herd watch` for this workspace (does not touch Claude Code sessions — those die with their terminal) |

What happens, in order:

1. `herd watch` is spawned in the background with `nohup`-style detach: pidfile at `.hyperherd/watch.pid`, log at `.hyperherd/watch.log`. If a watch is already running for this workspace, the second start is skipped with a note.
2. Project-scope `.claude/settings.local.json` is updated with allow-rules covering the agent's tool surface (`Bash(herd *)`, edits to `hyperherd.yaml`, reads/writes under `.hyperherd/`). Existing unrelated entries are preserved; entries already present are deduped. Skip with `--no-auto-allow`.
3. The initial prompt (`Use the hyperherd-monitor skill to manage the sweep at <workspace>… /loop` (dynamic, self-pacing via `ScheduleWakeup`)) is written to `.hyperherd/monitor-prompt.txt` as an audit trail.
4. `herd monitor` `exec`s `claude "<prompt>"` — an interactive Claude Code session that's already executing the prompt as its first user turn. No paste step. The skill takes over from there: setup interview, phased rollout, per-tick monitoring with adaptive cadence.
5. When you close your terminal, Claude Code exits — but `herd watch` keeps running in the background. To detach without losing the Claude session, wrap the whole thing in `tmux new -s monitor 'herd monitor'`.

To shut everything down, run `herd monitor --stop` (kills `watch`); the Claude session is already gone if you closed its terminal.

The auto-allow rules are scoped to **this workspace only** and only cover `herd` subcommands, the workspace's `hyperherd.yaml`, and files under `.hyperherd/`. If you'd rather grant permissions interactively, pass `--no-auto-allow` — but expect to approve a tool call every few minutes as the agent ticks.

If `claude` isn't on PATH, `herd monitor` prints the prompt and exits with an error — you can still paste the prompt into an existing Claude Code session manually.

## `herd msg`

Post a free-text message to the same webhook `herd watch` is using.

```bash
herd msg [-w WORKSPACE] MESSAGE...
```

Useful for narrating a sweep alongside the daemon's automatic events — "kicked off the rerun", "ignore the next failure, that's me killing trial 7", "out for the night, ping me on completion". The message is rendered with the same `[<sweep>] ...` prefix as daemon events and goes to whichever channel `watch.webhook`/`watch.format` resolve to (Slack, Discord, ntfy, or the zero-config per-workspace ntfy topic).

```bash
cd ~/sweeps/mnist_sweep
herd msg "rerunning failed trials with --force"

# From outside the workspace:
herd msg -w ~/sweeps/mnist_sweep "heads up — moving to a different partition"
```

Unlike `herd watch`, delivery errors are surfaced (non-zero exit, error printed) so you know the message didn't actually go out.

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
