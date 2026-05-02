# Launcher script

The launcher script is a user-provided bash script. HyperHerd does **not** manage your container runtime, environment modules, conda environment, or any other setup — that's your launcher's job.

## Contract

1. HyperHerd calls your launcher as: `bash <launcher_path> "<overrides>"`.
2. The first argument (`$1`) is a space-separated string of `name=value` pairs (e.g. `experiment_name=lr-0.001 learning_rate=0.001 optimizer=adam`).
3. Your script invokes the training command, passing those overrides through to whatever CLI your trainer exposes. [Hydra](https://hydra.cc/) is the recommended trainer harness because it consumes this format natively (`python train.py $OVERRIDES`), but you're free to parse the string however you want — split on spaces, transform into `--key value` flags, write a YAML overlay, whatever fits.
4. SLURM environment variables (`$SLURM_JOB_ID`, `$SLURM_ARRAY_TASK_ID`, …) are available, plus:
    - `$HYPERHERD_WORKSPACE` — absolute path to the workspace
    - `$HYPERHERD_TRIAL_ID` — array task index
    - `$HYPERHERD_EXPERIMENT_NAME` — auto-generated experiment name
5. Exit code 0 means success; nonzero means failure.

!!! note "Hydra-specific flag"
    `herd test --cfg-job` appends `--cfg job` to the overrides so Hydra prints the resolved config and exits without running training. The base `herd test` command is launcher-agnostic — only the `--cfg-job` flag is Hydra-specific.

## Examples

=== "Apptainer / Singularity (Hydra)"

    ```bash
    #!/bin/bash
    set -euo pipefail

    OVERRIDES="$1"
    CONTAINER="/path/to/your/container.sif"
    BINDS="/scratch:/scratch,/home/$USER:/home/$USER"

    apptainer exec --nv --bind "$BINDS" "$CONTAINER" \
        python train.py $OVERRIDES
    ```

=== "Conda (Hydra)"

    ```bash
    #!/bin/bash
    set -euo pipefail

    OVERRIDES="$1"

    source /opt/conda/etc/profile.d/conda.sh
    conda activate myenv

    python train.py $OVERRIDES
    ```

=== "Enroot / Pyxis"

    ```bash
    #!/bin/bash
    set -euo pipefail

    OVERRIDES="$1"
    IMAGE="nvcr.io/nvidia/pytorch:24.01-py3"

    srun --container-image="$IMAGE" \
         --container-mounts="/scratch:/scratch" \
         python train.py $OVERRIDES
    ```

=== "Non-Hydra Python trainer (`parse_overrides`)"

    If your trainer is a Python script and you don't want to write your own argparse, HyperHerd ships a tiny helper that turns the override string into a dict:

    ```bash
    # launch.sh
    #!/bin/bash
    set -euo pipefail
    python train.py "$1"
    ```

    ```python
    # train.py
    from hyperherd import parse_overrides, log_result

    params = parse_overrides()        # reads sys.argv[1]
    lr = params["lr"]                 # int/float/bool/None coerced from string
    optimizer = params["optimizer"]
    exp_name = params["experiment_name"]
    ...
    log_result("test_accuracy", acc)
    ```

    This only helps when your launcher delegates to Python — for R / Julia / MATLAB / raw bash launchers, parse the `name=value` string yourself in the launcher's language.

=== "Non-Hydra trainer (argparse-style flags)"

    For a trainer that takes `--key value` flags, translate `name=value` into flags before invoking it:

    ```bash
    #!/bin/bash
    set -euo pipefail

    OVERRIDES="$1"

    # Translate "key=value key2=value2" -> "--key value --key2 value2"
    FLAGS=()
    for kv in $OVERRIDES; do
        FLAGS+=("--${kv%%=*}" "${kv#*=}")
    done

    python train.py "${FLAGS[@]}"
    ```

## Activate your environment inside the launcher

The single most common cause of "it works on the login node, fails inside the SLURM job" is the launcher invoking `python` by name and getting a different interpreter on the worker than on your shell.

**Why it happens.** SLURM's default `--export=ALL` copies environment variables to the job, but it does not re-run your shell startup. On the worker, `bash launch.sh` is a non-interactive shell; depending on the cluster's bash config it may or may not source `.bashrc`, and even when it does, anything that activates a venv interactively (the `uv` shell hooks, `conda init`, `source .venv/bin/activate` typed at the prompt) won't have run. Whatever first `python` is on PATH wins — usually the system one, which doesn't have your packages.

**The footprint of this bug.** Trial fails immediately with `ModuleNotFoundError: No module named 'hydra'` (or torch, lightning, etc.) even though `python -c "import hydra"` works fine in the terminal you ran `herd run` from.

**Fixes** (any of these, pick the one that matches your stack):

=== "uv"

    ```bash
    #!/bin/bash
    set -euo pipefail

    OVERRIDES="$1"
    cd "$(dirname "$0")"
    uv run python train.py $OVERRIDES
    ```

    `uv run` finds the project's `.venv` automatically (via `pyproject.toml` / `uv.lock`). Requires `uv` to be on the worker's PATH — usually true when it lives in `~/.local/bin` and your home dir is mounted on compute nodes.

=== "venv (manual activate)"

    ```bash
    #!/bin/bash
    set -euo pipefail

    source /absolute/path/to/.venv/bin/activate
    OVERRIDES="$1"
    cd "$(dirname "$0")"
    python train.py $OVERRIDES
    ```

=== "conda"

    ```bash
    #!/bin/bash
    set -euo pipefail

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate myenv
    OVERRIDES="$1"
    cd "$(dirname "$0")"
    python train.py $OVERRIDES
    ```

=== "absolute interpreter"

    ```bash
    #!/bin/bash
    set -euo pipefail

    OVERRIDES="$1"
    cd "$(dirname "$0")"
    /absolute/path/to/.venv/bin/python train.py $OVERRIDES
    ```

    No PATH games — but the path is hardcoded, so the launcher isn't portable across machines.

=== "module + apptainer"

    ```bash
    #!/bin/bash
    set -euo pipefail

    module load apptainer  # if your cluster uses environment modules
    OVERRIDES="$1"
    apptainer exec --nv /path/to/container.sif \
        python train.py $OVERRIDES
    ```

    Inside the container the environment is hermetic — host PATH doesn't matter.

If your launcher works on the login node but fails inside SLURM with import errors, this is almost always the cause.

## Idempotency

`herd run` is idempotent — rerunning it resubmits only `ready`/`failed`/`cancelled` trials. For this to be useful, **your training script must also be idempotent**:

- **Checkpoint on a deterministic path.** Use `$HYPERHERD_EXPERIMENT_NAME` or `$HYPERHERD_TRIAL_ID` to construct a unique, stable output directory.
- **Resume from checkpoint.** On startup, check if a checkpoint exists and resume.
- **Don't fail on existing output.** Handle pre-existing output directories gracefully.

```python
import os
exp_name = os.environ.get("HYPERHERD_EXPERIMENT_NAME", "default")
output_dir = f"./outputs/{exp_name}"
```

## Compute nodes without Python

The default sbatch script invokes `python -m hyperherd resolve-overrides …` on the compute node to look up overrides from the manifest. If `python` isn't available on the bare compute node (only inside your container), you can read the manifest directly with `jq`:

```bash
#!/bin/bash
set -euo pipefail

MANIFEST=".hyperherd/manifest.json"
TASK_ID="$SLURM_ARRAY_TASK_ID"

OVERRIDES=$(jq -r --argjson id "$TASK_ID" '
  .[] | select(.index == $id) | .params | to_entries | map("\(.key)=\(.value)") | join(" ")
' "$MANIFEST")

EXPERIMENT_NAME=$(jq -r --argjson id "$TASK_ID" '
  .[] | select(.index == $id) | .experiment_name
' "$MANIFEST")

OVERRIDES="experiment_name=$EXPERIMENT_NAME $OVERRIDES data.root=/scratch/imagenet"

CONTAINER="/shared/containers/pytorch-24.01.sif"
apptainer exec --nv --bind "/scratch:/scratch" "$CONTAINER" \
    python train.py $OVERRIDES
```
