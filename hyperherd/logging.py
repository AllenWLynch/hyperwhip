"""Lightweight result logging for HyperHerd trials.

Usage from within a training script:

    from hyperherd.logging import log_result

    log_result("test_accuracy", 0.95)
    log_result("test_loss", 0.12)
    log_result("epochs_completed", 50)

Results are written to .hyperherd/results/<trial_id>.json in the workspace.
The workspace and trial ID are resolved from HYPERHERD_WORKSPACE and
HYPERHERD_TRIAL_ID environment variables (set automatically by mush).
"""

import json
import os

from hyperherd.manifest import WORKSPACE_DIR

RESULTS_DIR = "results"


def _results_dir() -> str:
    workspace = os.environ.get("HYPERHERD_WORKSPACE")
    if not workspace:
        raise RuntimeError(
            "HYPERHERD_WORKSPACE not set. "
            "log_result() must be called from within a HyperHerd trial "
            "(launched via 'mush launch' or 'mush test')."
        )
    return os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR)


def _results_path() -> str:
    trial_id = os.environ.get("HYPERHERD_TRIAL_ID")
    if trial_id is None:
        raise RuntimeError(
            "HYPERHERD_TRIAL_ID not set. "
            "log_result() must be called from within a HyperHerd trial."
        )
    return os.path.join(_results_dir(), f"{trial_id}.json")


def log_result(name: str, value) -> None:
    """Log a named metric for the current trial.

    Can be called multiple times — results accumulate in a single JSON file
    per trial. If called with the same name twice, the later value overwrites.

    Args:
        name: Metric name (e.g. "test_accuracy", "final_loss").
        value: Metric value (must be JSON-serializable).
    """
    path = _results_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing results if any
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[name] = value

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_trial_results(workspace: str, trial_id: int) -> dict:
    """Load results for a specific trial. Returns empty dict if no results."""
    path = os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR, f"{trial_id}.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_all_results(workspace: str) -> dict:
    """Load results for all trials. Returns {trial_id: {metric: value, ...}}."""
    results_dir = os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR)
    if not os.path.isdir(results_dir):
        return {}
    all_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            trial_id = int(fname.replace(".json", ""))
            with open(os.path.join(results_dir, fname), "r") as f:
                all_results[trial_id] = json.load(f)
    return all_results
