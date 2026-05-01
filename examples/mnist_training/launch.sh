#!/bin/bash
set -euo pipefail

OVERRIDES="$1"

echo "=== HyperHerd MNIST Launcher ==="
echo "  HYPERHERD_TRIAL_ID:          ${HYPERHERD_TRIAL_ID:-unset}"
echo "  HYPERHERD_EXPERIMENT_NAME:   ${HYPERHERD_EXPERIMENT_NAME:-unset}"
echo ""

cd "$(dirname "$0")"
python train.py $OVERRIDES