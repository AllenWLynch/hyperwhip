#!/bin/bash
set -euo pipefail

# SLURM workers don't re-run your shell startup, so any venv activated only
# in your interactive session won't be active here. Pick whichever line below
# matches your stack — see docs/launcher.md "Activate your environment".
#
#   uv run:   uv run python train.py $OVERRIDES
#   venv:     source /absolute/path/.venv/bin/activate; python train.py $OVERRIDES
#   conda:    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate <env>; python train.py $OVERRIDES
#   abs path: /absolute/path/.venv/bin/python train.py $OVERRIDES

OVERRIDES="$1"

echo "=== HyperHerd MNIST Launcher ==="
echo "  HYPERHERD_TRIAL_ID:          ${HYPERHERD_TRIAL_ID:-unset}"
echo "  HYPERHERD_EXPERIMENT_NAME:   ${HYPERHERD_EXPERIMENT_NAME:-unset}"
echo ""

cd "$(dirname "$0")"
uv run python train.py $OVERRIDES
