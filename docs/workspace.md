# Workspace layout

After `herd run`, the workspace directory contains:

```
my_experiment/
  hyperherd.yaml
  launch.sh
  .hyperherd/
    manifest.json       # Trial records: {index, params, experiment_name, status}
    job_ids.json        # Records of submitted SLURM jobs
    job.sbatch          # The generated sbatch script
    logs/
      0.out, 0.err      # stdout / stderr for array task 0
      1.out, 1.err
      ...
    results/
      0.json            # Logged metrics for trial 0 (from log_result())
      1.json
      ...
```

| File | Purpose |
|------|---------|
| `manifest.json` | Authoritative mapping of array index → parameter values + experiment name + status. Don't edit manually. |
| `job_ids.json` | Records each `herd run` submission with its SLURM job ID and the indices it covered. Used to reconcile status across resubmissions. |
| `job.sbatch` | The generated SLURM script. Inspect it to verify directives. |
| `logs/` | SLURM-captured stdout/stderr. `herd status` reads the last line of each `.out`. |
| `results/` | Per-trial JSON metrics written by `log_result()`. Read by `herd res`. |

## Generated SLURM script

For reference, HyperHerd generates a batch script roughly like this:

```bash
#!/bin/bash
#SBATCH --job-name=hyperherd_resnet_sweep
#SBATCH --array=0-11
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=<workspace>/.hyperherd/logs/%a.out
#SBATCH --error=<workspace>/.hyperherd/logs/%a.err
#SBATCH --open-mode=append
#SBATCH --export=ALL

# Export HyperHerd environment variables
export HYPERHERD_WORKSPACE="<workspace>"
export HYPERHERD_SWEEP_NAME="resnet_sweep"
export HYPERHERD_TRIAL_ID="$SLURM_ARRAY_TASK_ID"

# Per-trial values (HYPERHERD_TRIAL_NAME + OVERRIDES) are baked into the
# generated script via a `case "$SLURM_ARRAY_TASK_ID" in ... esac` block.
HYPERHERD_TRIAL_NAME="lr-0.001_opt-adam_bs-64"  # set per-task
export HYPERHERD_TRIAL_NAME
OVERRIDES="experiment_name=$HYPERHERD_TRIAL_NAME learning_rate=0.001 optimizer=adam batch_size=64 data.root=/scratch/imagenet trainer.max_epochs=90"

# Invoke the user's launcher script
bash "<workspace>/launch.sh" "$OVERRIDES"
```

The actual generated script bakes the per-trial values directly into a `case` statement at submission time, so neither `python` nor the `hyperherd` package needs to be available on the compute node — only `bash`.

## Re-running and reconciliation

`herd run` against an existing workspace:

1. Loads the existing manifest.
2. Generates the new manifest from the current `hyperherd.yaml`.
3. Diffs them.
4. **New trials** are appended.
5. **Removed trials**:
    - If their status is in (`submitted`, `queued`, `running`, `completed`), `herd run` refuses unless you pass `-f` (in which case they're kept as orphans for traceability).
    - Otherwise they're dropped.
6. Submits all `ready` / `failed` / `cancelled` trials.

This means you can edit your sweep mid-experiment (add a new parameter value, tighten a condition) without losing already-completed trials.
