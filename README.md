# HyperWhip

Launch and monitor hyperparameter optimization job arrays on SLURM.

HyperWhip takes a YAML configuration file describing your hyperparameter search space and submits it as a SLURM job array. Each array task runs one parameter combination through your training script via [Hydra](https://hydra.cc/) overrides. A user-provided launcher script handles container setup, environment modules, or any other runtime configuration.

## Installation

### From source (recommended for development)

```bash
git clone <repo-url> && cd hyperwhip
pip install -e .
```

### From the repository directly

```bash
pip install git+<repo-url>
```

### Verify installation

```bash
hyperwhip --help
```

This should print the available subcommands: `init`, `launch`, `monitor`, `clean`.

### Dependencies

- Python >= 3.8
- [PyYAML](https://pyyaml.org/) (installed automatically)
- A SLURM cluster with `sbatch`, `sacct`, `squeue`, and `scancel` available on the submission host

## Quick Start

### 1. Scaffold a new experiment

```bash
hyperwhip init my_experiment --partition gpu --gres gpu:1
```

This creates `my_experiment/hyperwhip.yaml` and `my_experiment/launch.sh` with sensible defaults. Edit both files to match your setup:

- **hyperwhip.yaml** — define your parameters, search mode, SLURM resources, and constraints
- **launch.sh** — set up your container, conda environment, or module loads

### 2. Preview and launch

```bash
# Preview first (runs preflight checks + prints trial list):
hyperwhip launch my_experiment/hyperwhip.yaml --dry-run

# Submit:
hyperwhip launch my_experiment/hyperwhip.yaml
```

Preflight checks run automatically before every launch and dry-run. They verify your launcher script exists and is executable, the workspace is writable, parameter definitions are valid, constraint references match defined parameters, and (if on a SLURM node) the partition exists.

### 3. Monitor

```bash
hyperwhip monitor my_experiment/hyperwhip.yaml
```

### 4. Resubmit failures

```bash
# Re-running launch only resubmits pending/failed trials:
hyperwhip launch my_experiment/hyperwhip.yaml
```

### 5. Clean up

```bash
hyperwhip clean my_experiment/hyperwhip.yaml --all
```

## Documentation

See [docs/configuration.md](docs/configuration.md) for the full configuration reference, launcher script examples, constraint system, and search mode details.
