"""SLURM interaction: sbatch submission, status queries, job cancellation."""

import dataclasses
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

from hyperherd.config import Config
from hyperherd import manifest


def generate_sbatch_script(
    config: Config, indices: List[int], max_concurrent: Optional[int] = None
) -> str:
    """Generate a SLURM batch script for the job array.

    If `max_concurrent` is given it overrides `config.slurm.max_concurrent`.
    """
    ws = manifest.workspace_path(config.workspace)
    log_dir = manifest.logs_path(config.workspace)

    # Build array spec (e.g. "0-49" or "0,2,5,7-10")
    array_spec = _indices_to_array_spec(indices)
    throttle = max_concurrent if max_concurrent is not None else config.slurm.max_concurrent
    if throttle is not None:
        array_spec = f"{array_spec}%{throttle}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=hyperherd_{config.name}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --partition={config.slurm.partition}",
        f"#SBATCH --time={config.slurm.time}",
        f"#SBATCH --mem={config.slurm.mem}",
        f"#SBATCH --cpus-per-task={config.slurm.cpus_per_task}",
    ]

    if config.slurm.gres:
        lines.append(f"#SBATCH --gres={config.slurm.gres}")

    lines.extend([
        f"#SBATCH --output={log_dir}/%a.out",
        f"#SBATCH --error={log_dir}/%a.err",
        # Append on resubmission so a trial's full history is preserved.
        # The divider below makes each run easy to find in the file.
        "#SBATCH --open-mode=append",
    ])

    for arg in config.slurm.extra_args:
        lines.append(f"#SBATCH {arg}")

    # Build static overrides for the resolve command
    static_flag = ""
    if config.hydra.static_overrides:
        escaped = " ".join(config.hydra.static_overrides)
        static_flag = f' --static "{escaped}"'

    lines.extend([
        "",
        "# Run divider — visible in both stdout and stderr after append",
        '_HH_DIVIDER="==== HyperHerd run: job ${SLURM_JOB_ID} '
        'array-task ${SLURM_ARRAY_TASK_ID} $(date -Iseconds) ===="',
        'printf "\\n%s\\n\\n" "$_HH_DIVIDER"',
        'printf "\\n%s\\n\\n" "$_HH_DIVIDER" >&2',
        "",
        "# Export HyperHerd environment variables",
        f'export HYPERHERD_WORKSPACE="{config.workspace}"',
        "export HYPERHERD_TRIAL_ID=\"$SLURM_ARRAY_TASK_ID\"",
        f'export HYPERHERD_EXPERIMENT_NAME=$(python -m hyperherd resolve-name '
        f'"{ws}/{manifest.MANIFEST_FILE}" "$SLURM_ARRAY_TASK_ID")',
        "",
        "# Resolve Hydra overrides for this array task (includes experiment_name=...)",
        f'OVERRIDES=$(python -m hyperherd resolve-overrides "{ws}/{manifest.MANIFEST_FILE}" '
        f'"$SLURM_ARRAY_TASK_ID"{static_flag})',
        "",
        "# Invoke the user's launcher script",
        f'bash "{config.launcher}" "$OVERRIDES"',
    ])

    return "\n".join(lines) + "\n"


def _indices_to_array_spec(indices: List[int]) -> str:
    """Convert a list of indices to a compact SLURM array spec.

    Examples:
        [0, 1, 2, 3] -> "0-3"
        [0, 1, 3, 5, 6, 7] -> "0-1,3,5-7"
    """
    if not indices:
        raise ValueError("No indices to submit")

    indices = sorted(set(indices))
    ranges = []
    start = indices[0]
    end = indices[0]

    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append((start, end))
            start = i
            end = i
    ranges.append((start, end))

    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(str(s))
        else:
            parts.append(f"{s}-{e}")

    return ",".join(parts)


def submit_job(config: Config, script_content: str, dry_run: bool = False) -> Optional[str]:
    """Submit a job array via sbatch. Returns the SLURM job ID, or None for dry run."""
    sbatch_file = manifest.sbatch_path(config.workspace)

    with open(sbatch_file, "w") as f:
        f.write(script_content)
    os.chmod(sbatch_file, 0o755)

    if dry_run:
        return None

    result = subprocess.run(
        ["sbatch", sbatch_file],
        capture_output=True,
        text=True,
        cwd=config.workspace,
    )

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")

    return match.group(1)


def query_job_status(job_ids: List[str]) -> Dict[Tuple[str, int], str]:
    """Query SLURM for the status of job array tasks.

    Returns a dict of (job_id, array_index) -> status string.
    Status values: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.
    """
    return {k: v.state for k, v in query_job_stats(job_ids).items()}


@dataclasses.dataclass
class JobStats:
    """Accounting info for one array-task, fused across the parent + .batch step rows."""
    state: str = "UNKNOWN"
    elapsed: str = ""
    max_rss: str = ""        # peak resident memory (e.g. '1234K', '2.3G')
    ave_rss: str = ""
    req_mem: str = ""
    max_vm: str = ""


def query_job_stats(job_ids: List[str]) -> Dict[Tuple[str, int], JobStats]:
    """Query SLURM accounting for state + memory/runtime stats per array-task.

    sacct returns one row for the parent (`12345_0`) carrying State/Elapsed,
    and a separate row for the batch step (`12345_0.batch`) carrying
    MaxRSS/AveRSS/MaxVMSize. We fuse both into one JobStats record per
    (jid, idx).
    """
    if not job_ids:
        return {}

    job_spec = ",".join(job_ids)
    result = subprocess.run(
        [
            "sacct",
            "-j", job_spec,
            "--format=JobID,State,Elapsed,MaxRSS,AveRSS,ReqMem,MaxVMSize",
            "--noheader",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # sacct unavailable; degrade to squeue (state only).
        return {k: JobStats(state=v) for k, v in _query_squeue(job_ids).items()}

    stats: Dict[Tuple[str, int], JobStats] = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id_str = parts[0]
        state = parts[1].split()[0] if parts[1] else "UNKNOWN"
        elapsed = parts[2] if len(parts) > 2 else ""
        max_rss = parts[3] if len(parts) > 3 else ""
        ave_rss = parts[4] if len(parts) > 4 else ""
        req_mem = parts[5] if len(parts) > 5 else ""
        max_vm = parts[6] if len(parts) > 6 else ""

        # Parent row: "12345_0"
        match = re.match(r"(\d+)_(\d+)$", job_id_str)
        if match:
            key = (match.group(1), int(match.group(2)))
            entry = stats.setdefault(key, JobStats())
            entry.state = state
            entry.elapsed = elapsed or entry.elapsed
            entry.req_mem = req_mem or entry.req_mem
            continue

        # Step row: "12345_0.batch" (carries memory stats)
        match = re.match(r"(\d+)_(\d+)\.batch$", job_id_str)
        if match:
            key = (match.group(1), int(match.group(2)))
            entry = stats.setdefault(key, JobStats())
            if max_rss:
                entry.max_rss = max_rss
            if ave_rss:
                entry.ave_rss = ave_rss
            if max_vm:
                entry.max_vm = max_vm
            continue

        # Compact range form: "12345_[0-10]" — cancelled before any task started.
        match = re.match(r"(\d+)_\[(.+)\]$", job_id_str)
        if match:
            jid = match.group(1)
            for idx in _parse_array_range(match.group(2)):
                stats.setdefault((jid, idx), JobStats()).state = state

    return stats


def _parse_array_range(spec: str) -> List[int]:
    """Parse a SLURM array range spec like '0-10' or '0-3,5,7-9' into indices."""
    indices = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def _query_squeue(job_ids: List[str]) -> Dict[Tuple[str, int], str]:
    """Fallback: query squeue for running/pending jobs."""
    job_spec = ",".join(job_ids)
    result = subprocess.run(
        [
            "squeue",
            "-j", job_spec,
            "--format=%i %t",
            "--noheader",
        ],
        capture_output=True,
        text=True,
    )

    statuses = {}
    if result.returncode != 0:
        return statuses

    state_map = {
        "PD": "PENDING",
        "R": "RUNNING",
        "CG": "COMPLETING",
        "CD": "COMPLETED",
        "F": "FAILED",
        "CA": "CANCELLED",
        "TO": "TIMEOUT",
    }

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        job_id_str = parts[0]
        state_code = parts[1]

        match = re.match(r"(\d+)_(\d+)", job_id_str)
        if match:
            jid = match.group(1)
            array_idx = int(match.group(2))
            statuses[(jid, array_idx)] = state_map.get(state_code, state_code)

    return statuses


def cancel_jobs(job_ids: List[str]) -> None:
    """Cancel SLURM jobs via scancel."""
    if not job_ids:
        return
    for jid in job_ids:
        subprocess.run(["scancel", jid], capture_output=True, text=True)


def cancel_array_task(job_id: str, array_index: int) -> None:
    """Cancel a single task within a SLURM job array via scancel <jid>_<idx>."""
    subprocess.run(
        ["scancel", f"{job_id}_{array_index}"], capture_output=True, text=True
    )


def get_log_tail(base: str, index: int, lines: int = 1) -> str:
    """Read the last N lines from a task's stdout log."""
    log_file = os.path.join(manifest.logs_path(base), f"{index}.out")
    if not os.path.isfile(log_file):
        return ""
    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            tail = all_lines[-lines:] if all_lines else []
            return "".join(tail).strip()
    except (OSError, UnicodeDecodeError):
        return "<unreadable>"
