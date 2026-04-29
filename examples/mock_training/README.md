# Mock Training Example

A self-contained example that demonstrates HyperHerd's full workflow without a real ML framework or SLURM cluster.

## What it does

`train.py` is a Hydra-configured mock training script that:

- **Simulates training** with `time.sleep()` per epoch
- **Writes checkpoints** after each epoch (`outputs/<experiment_name>/checkpoint.json`)
- **Randomly fails** with configurable probability (`failure_probability=0.3`)
- **Resumes from checkpoints** when re-invoked with the same parameters (idempotent)
- **Writes final results** as `outputs/<experiment_name>/results.json`

The fake accuracy metric varies deterministically based on hyperparameters (lower LR converges slower but better, Adam gets a bonus, larger batch sizes are slightly worse).

## Prerequisites

```bash
pip install hydra-core omegaconf
```

## Running locally (without SLURM)

```bash
cd examples/mock_training

# Preview the sweep:
hyperherd launch hyperherd.yaml --dry-run

# You can test a single trial manually:
python train.py learning_rate=0.001 optimizer=adam batch_size=64 experiment_name=test_run

# If it fails, run the same command — it resumes from checkpoint:
python train.py learning_rate=0.001 optimizer=adam batch_size=64 experiment_name=test_run
```

## Running on SLURM

```bash
cd examples/mock_training

# Submit all 12 trials (3 LR x 2 optimizer x 2 batch_size):
hyperherd launch hyperherd.yaml

# Monitor progress:
hyperherd monitor hyperherd.yaml

# Some trials will fail randomly. Re-launch to resubmit only the failed ones:
hyperherd launch hyperherd.yaml

# Repeat until all trials complete.

# Clean up:
hyperherd clean hyperherd.yaml --all
```

## What to observe

1. **Dry-run** shows 12 trials, each with an `experiment_name` like `lr=0.01_opt=adam_bs=32`
2. **First launch** submits all 12 — ~30% will fail randomly per epoch
3. **Monitor** shows a mix of COMPLETED, FAILED, and RUNNING statuses
4. **Re-launch** only resubmits the failed trials — they resume from their last checkpoint
5. **After a few re-launches**, all 12 trials complete
6. **Results** are in `outputs/<experiment_name>/results.json` for each trial
