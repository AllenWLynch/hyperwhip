"""Pre-flight validation checks run before launch and dry-run."""

import os
import subprocess
from typing import List

from hyperwhip.config import Config


class PreflightError(Exception):
    """Raised when a preflight check fails."""


class PreflightWarning:
    """A non-fatal preflight issue."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


def run_preflight(config: Config, strict: bool = False) -> List[PreflightWarning]:
    """Run all preflight checks. Raises PreflightError on fatal issues.

    Returns a list of non-fatal warnings.
    """
    warnings = []

    _check_launcher(config)
    _check_workspace_writable(config)
    _check_parameters_nonempty(config)
    _check_constraint_refs(config)

    if config.search.mode == "axes":
        _check_axes_defaults(config)

    warnings.extend(_check_partition(config))

    return warnings


def _check_launcher(config: Config) -> None:
    """Verify the launcher script exists and is executable."""
    if not config.launcher:
        raise PreflightError(
            "No launcher script specified. Set the 'launcher' field in your config."
        )
    if not os.path.isfile(config.launcher):
        raise PreflightError(
            f"Launcher script not found: {config.launcher}\n"
            f"  Create it or fix the 'launcher' path in your config."
        )
    if not os.access(config.launcher, os.X_OK):
        raise PreflightError(
            f"Launcher script is not executable: {config.launcher}\n"
            f"  Run: chmod +x {config.launcher}"
        )


def _check_workspace_writable(config: Config) -> None:
    """Verify we can write to the workspace parent directory."""
    ws = config.workspace
    # If workspace already exists, check it's writable
    if os.path.isdir(ws):
        if not os.access(ws, os.W_OK):
            raise PreflightError(
                f"Workspace directory is not writable: {ws}"
            )
        return

    # Otherwise check the parent is writable (we'll create workspace on launch)
    parent = os.path.dirname(ws)
    if not parent:
        parent = "."
    if not os.path.isdir(parent):
        raise PreflightError(
            f"Workspace parent directory does not exist: {parent}\n"
            f"  Create it or change the 'workspace' path in your config."
        )
    if not os.access(parent, os.W_OK):
        raise PreflightError(
            f"Workspace parent directory is not writable: {parent}"
        )


def _check_parameters_nonempty(config: Config) -> None:
    """Verify at least one parameter is defined and all have valid values."""
    if not config.parameters:
        raise PreflightError("No parameters defined. Add at least one parameter to sweep over.")

    for param in config.parameters:
        if param.type == "discrete":
            if not param.values:
                raise PreflightError(
                    f"Parameter '{param.name}': discrete type has empty values list."
                )
        elif param.type == "continuous":
            if param.low >= param.high:
                raise PreflightError(
                    f"Parameter '{param.name}': low ({param.low}) must be less than high ({param.high})."
                )
            if param.steps < 1:
                raise PreflightError(
                    f"Parameter '{param.name}': steps must be >= 1, got {param.steps}."
                )
            if param.scale == "log" and param.low <= 0:
                raise PreflightError(
                    f"Parameter '{param.name}': log scale requires low > 0, got {param.low}."
                )


def _check_constraint_refs(config: Config) -> None:
    """Verify constraint 'when', 'exclude', and 'force' reference defined parameters."""
    param_names = {p.name for p in config.parameters}

    for constraint in config.constraints:
        for ref in constraint.when:
            if ref not in param_names:
                raise PreflightError(
                    f"Constraint '{constraint.name}': 'when' references unknown parameter '{ref}'.\n"
                    f"  Defined parameters: {sorted(param_names)}"
                )
        if constraint.exclude:
            for ref in constraint.exclude:
                if ref not in param_names:
                    raise PreflightError(
                        f"Constraint '{constraint.name}': 'exclude' references unknown parameter '{ref}'.\n"
                        f"  Defined parameters: {sorted(param_names)}"
                    )
        if constraint.force:
            for ref in constraint.force:
                if ref not in param_names:
                    raise PreflightError(
                        f"Constraint '{constraint.name}': 'force' references unknown parameter '{ref}'.\n"
                        f"  Defined parameters: {sorted(param_names)}"
                    )


def _check_axes_defaults(config: Config) -> None:
    """Verify axes-mode defaults reference valid parameter values."""
    if not config.search.defaults:
        raise PreflightError("axes mode requires 'search.defaults' for all parameters.")

    for param in config.parameters:
        default = config.search.defaults.get(param.name)
        if default is None:
            raise PreflightError(
                f"axes mode: no default value for parameter '{param.name}'."
            )
        if param.type == "discrete" and default not in param.values:
            raise PreflightError(
                f"axes mode: default value '{default}' for parameter '{param.name}' "
                f"is not in its values list: {param.values}"
            )
        if param.type == "continuous":
            try:
                val = float(default)
            except (TypeError, ValueError):
                raise PreflightError(
                    f"axes mode: default value '{default}' for continuous parameter "
                    f"'{param.name}' is not a number."
                )
            if val < param.low or val > param.high:
                raise PreflightError(
                    f"axes mode: default value {val} for parameter '{param.name}' "
                    f"is outside range [{param.low}, {param.high}]."
                )


def _check_partition(config: Config) -> List[PreflightWarning]:
    """Check if the SLURM partition exists. Returns warnings (non-fatal)."""
    warnings = []
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-p", config.slurm.partition, "--format=%P"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            warnings.append(PreflightWarning(
                f"SLURM partition '{config.slurm.partition}' not found or sinfo failed. "
                f"Verify the partition name is correct."
            ))
    except FileNotFoundError:
        warnings.append(PreflightWarning(
            "sinfo not found. Cannot verify SLURM partition. "
            "This is expected if you're not on a SLURM login node."
        ))
    except subprocess.TimeoutExpired:
        warnings.append(PreflightWarning(
            "sinfo timed out. Cannot verify SLURM partition."
        ))

    return warnings
