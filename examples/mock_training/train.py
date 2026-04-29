"""Mock training script for testing HyperHerd.

Simulates a multi-epoch training loop that:
- Sleeps to simulate computation
- Writes checkpoint files after each epoch
- Resumes from the last checkpoint (idempotent)
- Randomly fails with configurable probability
- Writes final metrics as a JSON artifact
"""

import json
import math
import os
import random
import sys
import time

import hydra
from omegaconf import DictConfig


def _deterministic_seed(cfg: DictConfig) -> int:
    """Derive a deterministic seed from the hyperparameters."""
    s = f"{cfg.learning_rate}_{cfg.optimizer}_{cfg.batch_size}"
    return hash(s) % (2**31)


def _simulate_metric(cfg: DictConfig, epoch: int) -> float:
    """Produce a fake metric that varies by hyperparameters and improves over epochs.

    Lower learning rates converge slower but to better final values.
    Adam-like optimizers get a bonus. Larger batch sizes are slightly worse.
    """
    seed = _deterministic_seed(cfg)
    rng = random.Random(seed + epoch)

    lr = cfg.learning_rate
    lr_factor = 1.0 - 0.1 * math.log10(lr + 1e-10)  # lower lr -> higher factor
    opt_bonus = 0.05 if cfg.optimizer in ("adam", "adamw") else 0.0
    bs_penalty = (cfg.batch_size - 32) * 0.0005
    epoch_progress = (epoch + 1) / cfg.num_epochs
    noise = rng.gauss(0, 0.02)

    accuracy = min(0.5 * epoch_progress * lr_factor + opt_bonus - bs_penalty + noise, 1.0)
    return round(accuracy, 4)


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, "checkpoint.json")


def _load_checkpoint(output_dir: str) -> int:
    """Load the last completed epoch from checkpoint. Returns -1 if none."""
    cp_path = _checkpoint_path(output_dir)
    if not os.path.isfile(cp_path):
        return -1
    with open(cp_path, "r") as f:
        data = json.load(f)
    return data.get("last_epoch", -1)


def _save_checkpoint(output_dir: str, epoch: int, metrics: dict) -> None:
    cp_path = _checkpoint_path(output_dir)
    with open(cp_path, "w") as f:
        json.dump({"last_epoch": epoch, "metrics": metrics}, f, indent=2)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    exp_name = cfg.get("experiment_name", "unknown")
    print(f"[HyperHerd Mock Training]")
    print(f"  experiment_name: {exp_name}")
    print(f"  learning_rate:   {cfg.learning_rate}")
    print(f"  optimizer:       {cfg.optimizer}")
    print(f"  batch_size:      {cfg.batch_size}")
    print(f"  output_dir:      {output_dir}")
    print(f"  num_epochs:      {cfg.num_epochs}")
    print(f"  failure_prob:    {cfg.failure_probability}")
    print()

    # Resume from checkpoint
    start_epoch = _load_checkpoint(output_dir) + 1
    # Load last metrics from checkpoint if resuming
    cp_path = _checkpoint_path(output_dir)
    metrics = {}
    if start_epoch > 0 and os.path.isfile(cp_path):
        with open(cp_path, "r") as f:
            cp_data = json.load(f)
        metrics = cp_data.get("metrics", {})
        print(f"  Resuming from epoch {start_epoch} (checkpoint found)")
    else:
        print(f"  Starting fresh")
    print()

    # Seed failure RNG using PID + time so each attempt has different failure outcomes.
    # (If failures were deterministic, retries would always fail at the same epoch.)
    fail_rng = random.Random(os.getpid() + int(time.time() * 1000))
    for epoch in range(start_epoch, cfg.num_epochs):
        # Check for random failure
        if fail_rng.random() < cfg.failure_probability:
            print(f"  epoch {epoch}/{cfg.num_epochs} - SIMULATED FAILURE!")
            sys.exit(1)

        # Simulate training
        print(f"  epoch {epoch}/{cfg.num_epochs} - training...", end="", flush=True)
        time.sleep(cfg.sleep_per_epoch)

        accuracy = _simulate_metric(cfg, epoch)
        loss = round(1.0 - accuracy, 4)
        metrics = {"epoch": epoch, "accuracy": accuracy, "loss": loss}
        print(f" accuracy={accuracy:.4f} loss={loss:.4f}")

        # Save checkpoint
        _save_checkpoint(output_dir, epoch, metrics)

    # Write final results
    results_path = os.path.join(output_dir, "results.json")
    final_results = {
        "experiment_name": exp_name,
        "learning_rate": cfg.learning_rate,
        "optimizer": cfg.optimizer,
        "batch_size": cfg.batch_size,
        "final_accuracy": metrics.get("accuracy", 0.0),
        "final_loss": metrics.get("loss", 1.0),
        "epochs_completed": cfg.num_epochs,
    }
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n  Training complete! Results saved to {results_path}")
    print(f"  Final accuracy: {metrics.get('accuracy', 0.0):.4f}")


if __name__ == "__main__":
    main()
