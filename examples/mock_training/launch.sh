#!/bin/bash
# Launcher for mock training example.
# No container needed — runs directly with the local Python environment.
set -euo pipefail

OVERRIDES="$1"

echo "=== HyperHerd Mock Launcher ==="
echo "  HYPERHERD_TRIAL_ID:          ${HYPERHERD_TRIAL_ID:-unset}"
echo "  HYPERHERD_EXPERIMENT_NAME:   ${HYPERHERD_EXPERIMENT_NAME:-unset}"
echo "  Overrides: $OVERRIDES"
echo ""

# Run the mock training script with Hydra overrides
cd "$(dirname "$0")"
python train.py $OVERRIDES
