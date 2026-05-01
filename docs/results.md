# Results & logging

HyperHerd ships a lightweight API for logging per-trial summary metrics from inside your training code. It's intended for **final / test metrics** — use wandb or similar for dense per-step training logs.

## Logging from training code

```python
from hyperherd import log_result

# Call at the end of training or evaluation:
log_result("test_accuracy", 0.95)
log_result("final_loss", 0.12)
log_result("epochs_completed", 50)
```

`log_result(name, value)` writes to `.hyperherd/results/<trial_id>.json` in the workspace. It resolves the write path from the `HYPERHERD_WORKSPACE` and `HYPERHERD_TRIAL_ID` environment variables, which are set automatically by `herd run`, `herd test`, and `herd local`.

- Can be called multiple times — metrics accumulate in a single JSON file per trial.
- Calling with the same metric name overwrites the previous value.
- Values must be JSON-serializable (numbers, strings, booleans, lists, dicts).

## Viewing results

```bash
herd res
```

Prints a TSV with every trial's parameters and logged metrics:

```
trial_id  experiment_name          learning_rate  optimizer  test_acc  test_loss
0         lr-0.01_opt-adam_bs-32   0.01           adam       0.92      0.31
1         lr-0.01_opt-adam_bs-64   0.01           adam       0.87      0.45
2         lr-0.01_opt-sgd_bs-32    0.01           sgd
3         lr-0.01_opt-sgd_bs-64    0.01           sgd
```

Trials without results show empty cells. Pipe to `column -t` for aligned display, or redirect to a file for pandas / Excel.

```bash
herd res > results.tsv
herd res | column -t -s $'\t' | less
```

## Custom downstream parsing

Each `.hyperherd/results/<trial_id>.json` is a flat JSON object you can read directly:

```python
import json, glob, os

workspace = "my_experiment"
results = {}
for path in glob.glob(f"{workspace}/.hyperherd/results/*.json"):
    trial_id = int(os.path.splitext(os.path.basename(path))[0])
    with open(path) as f:
        results[trial_id] = json.load(f)
```

For a richer join with the parameter manifest, read both:

```python
with open(f"{workspace}/.hyperherd/manifest.json") as f:
    manifest = {t["index"]: t for t in json.load(f)}
```
