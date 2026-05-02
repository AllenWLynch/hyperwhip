"""Tests for `--json` (agent) output across the read- and run-style commands.

Two layers:
  - the pure payload builders in `agent_output` (memory parsing, payload
    shapes); these are checked directly without driving the CLI
  - the CLI integration: each handler invoked with `json_output=True`,
    capturing stdout, asserting the shape and that the "human" chatter
    didn't leak in
"""

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import agent_output, manifest, slurm
from hyperherd.cli import (
    cmd_launch, cmd_status, cmd_results, cmd_snapshot, cmd_stats, cmd_stop,
    cmd_tail,
)


# --- payload-builder tests ---------------------------------------------------

class TestParseMemBytes(unittest.TestCase):
    def test_kilobyte_suffix(self):
        self.assertEqual(agent_output.parse_mem_bytes("1024K"), 1024 * 1024)

    def test_megabyte_suffix(self):
        self.assertEqual(agent_output.parse_mem_bytes("4M"), 4 * 1024 ** 2)

    def test_gigabyte_suffix(self):
        self.assertEqual(agent_output.parse_mem_bytes("2G"), 2 * 1024 ** 3)

    def test_fractional_value(self):
        self.assertEqual(agent_output.parse_mem_bytes("1.5G"), int(1.5 * 1024 ** 3))

    def test_no_suffix_treated_as_bytes(self):
        self.assertEqual(agent_output.parse_mem_bytes("512"), 512)

    def test_empty_returns_none(self):
        self.assertIsNone(agent_output.parse_mem_bytes(""))
        self.assertIsNone(agent_output.parse_mem_bytes("   "))

    def test_garbage_returns_none(self):
        self.assertIsNone(agent_output.parse_mem_bytes("eleventy"))


class TestParseElapsedSeconds(unittest.TestCase):
    def test_hms(self):
        self.assertEqual(agent_output.parse_elapsed_seconds("01:02:03"), 3723)

    def test_with_days(self):
        self.assertEqual(
            agent_output.parse_elapsed_seconds("2-01:00:00"),
            2 * 86400 + 3600,
        )

    def test_empty(self):
        self.assertIsNone(agent_output.parse_elapsed_seconds(""))

    def test_garbage(self):
        self.assertIsNone(agent_output.parse_elapsed_seconds("ten o'clock"))


# --- CLI integration helpers -------------------------------------------------

def _write_config(base: str) -> None:
    cfg = (
        "name: t\n"
        f"workspace: {base}\n"
        f"launcher: {os.path.join(base, 'launch.sh')}\n"
        "grid: all\n"
        "parameters:\n"
        "  lr:\n"
        "    type: discrete\n"
        "    abbrev: lr\n"
        "    values: [0.1, 0.01]\n"
        "slurm:\n"
        "  partition: p\n"
        "  time: '00:10:00'\n"
        "  mem: 1G\n"
        "  cpus_per_task: 1\n"
    )
    with open(os.path.join(base, "hyperherd.yaml"), "w") as f:
        f.write(cfg)
    with open(os.path.join(base, "launch.sh"), "w") as f:
        f.write("#!/bin/bash\n")
    os.chmod(os.path.join(base, "launch.sh"), 0o755)


class _CliJsonTestCase(unittest.TestCase):
    """Common scaffold: tmp workspace, two trials, json mode on."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _capture_json(self, fn, args):
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = fn(args)
        self.assertEqual(rc, 0, msg=f"handler returned {rc}")
        return rc, json.loads(buf.getvalue())


# --- run --dry-run --json (the agent-critical path) --------------------------

class TestDryRunJson(_CliJsonTestCase):
    """An agent uses `herd run --dry-run --json` to enumerate hparam
    combinations before deciding whether to submit. The payload must contain
    one entry per generated trial, each with its concrete params dict, and
    the sbatch script the agent could persist if it wanted to inspect it."""

    def _args(self):
        return argparse.Namespace(
            workspace=self.tmp,
            dry_run=True,
            max_concurrent=None,
            indices=None,
            force=False,
            json_output=True,
        )

    def test_dry_run_emits_machine_readable_combinations(self):
        _, payload = self._capture_json(cmd_launch, self._args())

        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["slurm_job_id"])
        self.assertIsNone(payload["sbatch_path"])
        self.assertEqual(payload["submitted_indices"], [0, 1])

        # Every trial carries its index and the actual hparam combination.
        self.assertEqual(len(payload["trials"]), 2)
        seen_lr = sorted(t["params"]["lr"] for t in payload["trials"])
        self.assertEqual(seen_lr, [0.01, 0.1])
        for t in payload["trials"]:
            self.assertEqual(t["status"], "ready")
            self.assertIn("experiment_name", t)

        # Full sbatch script available for the agent to inspect / save.
        self.assertIsNotNone(payload["sbatch_script"])
        self.assertIn("#SBATCH", payload["sbatch_script"])

    def test_dry_run_emits_no_human_chatter_on_stdout(self):
        # The agent parses stdout as JSON, so any rogue print() would corrupt
        # the parse. Capture stdout and round-trip through json.loads.
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            cmd_launch(self._args())
        # If anything else printed to stdout, json.loads would raise.
        body = buf.getvalue()
        json.loads(body)


# --- status --json -----------------------------------------------------------

class TestStatusJson(_CliJsonTestCase):
    def setUp(self):
        super().setUp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.01}],
            abbrevs={"lr": "lr"},
        )
        manifest.bulk_update_status(self.tmp, {0: "running", 1: "ready"})

    def test_status_json_shape(self):
        args = argparse.Namespace(workspace=self.tmp, json_output=True)
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
             mock.patch("hyperherd.slurm.get_log_tail", return_value=""):
            _, payload = self._capture_json(cmd_status, args)

        self.assertEqual(payload["totals"]["total"], 2)
        self.assertEqual(payload["totals"]["running"], 1)
        self.assertEqual(payload["totals"]["ready"], 1)
        self.assertEqual(len(payload["trials"]), 2)
        for t in payload["trials"]:
            self.assertIn(t["status"], {"running", "ready"})


# --- stats --json ------------------------------------------------------------

class TestStatsJson(_CliJsonTestCase):
    def setUp(self):
        super().setUp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.01}],
            abbrevs={"lr": "lr"},
        )
        manifest.record_job_submission(self.tmp, "12345", [0, 1])
        self.fake_stats = {
            ("12345", 0): slurm.JobStats(
                state="COMPLETED", elapsed="00:01:30",
                max_rss="1024K", ave_rss="512K", req_mem="1G",
            ),
            ("12345", 1): slurm.JobStats(
                state="RUNNING", elapsed="00:00:42",
            ),
        }

    def test_stats_json_emits_bytes_and_seconds(self):
        args = argparse.Namespace(workspace=self.tmp, index=None, json_output=True)
        with mock.patch("hyperherd.slurm.query_job_stats", return_value=self.fake_stats):
            _, payload = self._capture_json(cmd_stats, args)

        by_index = {t["index"]: t for t in payload["trials"]}
        completed = by_index[0]
        self.assertEqual(completed["slurm_state"], "COMPLETED")
        self.assertEqual(completed["elapsed_seconds"], 90)
        self.assertEqual(completed["max_rss_bytes"], 1024 * 1024)
        self.assertEqual(completed["req_mem_bytes"], 1024 ** 3)

        running = by_index[1]
        self.assertEqual(running["slurm_state"], "RUNNING")
        self.assertEqual(running["elapsed_seconds"], 42)
        self.assertIsNone(running["max_rss_bytes"])  # null, not 0
        self.assertIsNone(running["req_mem_bytes"])


# --- tail --json + --stdout/--stderr filters ---------------------------------

class TestTailJson(_CliJsonTestCase):
    def setUp(self):
        super().setUp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp, [{"lr": 0.1}], abbrevs={"lr": "lr"},
        )
        log_dir = manifest.logs_path(self.tmp)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "0.out"), "w") as f:
            f.write("epoch 1\nepoch 2\nepoch 3\n")
        with open(os.path.join(log_dir, "0.err"), "w") as f:
            f.write("warning: cuda OOM\n")

    def _args(self, **overrides):
        defaults = dict(
            workspace=self.tmp, index=0, lines=10, stream=None, json_output=True,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_default_includes_both_streams(self):
        _, payload = self._capture_json(cmd_tail, self._args())
        self.assertIn("stdout", payload["streams"])
        self.assertIn("stderr", payload["streams"])
        self.assertEqual(payload["streams"]["stdout"]["lines"], ["epoch 1", "epoch 2", "epoch 3"])
        self.assertEqual(payload["streams"]["stderr"]["lines"], ["warning: cuda OOM"])

    def test_stderr_only_filter(self):
        _, payload = self._capture_json(cmd_tail, self._args(stream="stderr"))
        self.assertEqual(set(payload["streams"]), {"stderr"})

    def test_stdout_only_filter(self):
        _, payload = self._capture_json(cmd_tail, self._args(stream="stdout"))
        self.assertEqual(set(payload["streams"]), {"stdout"})

    def test_missing_stream_marked_null(self):
        os.unlink(os.path.join(manifest.logs_path(self.tmp), "0.err"))
        _, payload = self._capture_json(cmd_tail, self._args())
        self.assertIsNone(payload["streams"]["stderr"]["lines"])
        # stdout still readable
        self.assertIsNotNone(payload["streams"]["stdout"]["lines"])


# --- stop --json -------------------------------------------------------------

class TestStopJson(_CliJsonTestCase):
    def setUp(self):
        super().setUp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.01}],
            abbrevs={"lr": "lr"},
        )
        manifest.record_job_submission(self.tmp, "12345", [0, 1])
        manifest.bulk_update_status(self.tmp, {0: "running", 1: "queued"})

    def test_all_emits_per_trial_records(self):
        args = argparse.Namespace(workspace=self.tmp, index=None, all=True, json_output=True)
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
             mock.patch("hyperherd.slurm.cancel_array_task") as cancel:
            _, payload = self._capture_json(cmd_stop, args)
        self.assertEqual(len(payload["cancelled"]), 2)
        for row in payload["cancelled"]:
            self.assertEqual(row["slurm_job_id"], "12345")
            self.assertIn(row["index"], {0, 1})
            self.assertIn(row["previous_status"], {"running", "queued"})
        self.assertEqual(cancel.call_count, 2)

    def test_no_live_trials_returns_empty_cancelled(self):
        manifest.bulk_update_status(self.tmp, {0: "completed", 1: "failed"})
        args = argparse.Namespace(workspace=self.tmp, index=None, all=True, json_output=True)
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            _, payload = self._capture_json(cmd_stop, args)
        self.assertEqual(payload["cancelled"], [])


# --- res --json --------------------------------------------------------------

class TestResultsJson(_CliJsonTestCase):
    def setUp(self):
        super().setUp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.01}],
            abbrevs={"lr": "lr"},
        )
        # Stash one trial's metric directly to disk.
        results_dir = os.path.join(self.tmp, ".hyperherd", "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "0.json"), "w") as f:
            json.dump({"test_acc": 0.95}, f)

    def test_results_json_includes_trials_without_metrics(self):
        args = argparse.Namespace(workspace=self.tmp, json_output=True)
        _, payload = self._capture_json(cmd_results, args)
        self.assertEqual(len(payload["trials"]), 2)
        by_index = {t["index"]: t for t in payload["trials"]}
        self.assertEqual(by_index[0]["metrics"], {"test_acc": 0.95})
        self.assertEqual(by_index[1]["metrics"], {})  # not silently dropped


# --- snapshot (the agent loop's per-tick payload) ----------------------------

class TestSnapshotJson(_CliJsonTestCase):
    """`herd snapshot` is the unified per-tick payload an agent reads. The
    test drives a workspace with a mix of completed / running / failed
    trials and a logged metric, and asserts the bundled document has each
    expected slice."""

    def setUp(self):
        super().setUp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.01}, {"lr": 0.001}],
            abbrevs={"lr": "lr"},
        )
        manifest.record_job_submission(self.tmp, "12345", [0, 1, 2])
        manifest.bulk_update_status(
            self.tmp, {0: "completed", 1: "running", 2: "failed"},
        )
        # one logged metric for the completed trial
        results_dir = os.path.join(self.tmp, ".hyperherd", "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "0.json"), "w") as f:
            json.dump({"test_acc": 0.95}, f)
        # stderr file for the failed trial
        log_dir = manifest.logs_path(self.tmp)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "2.err"), "w") as f:
            f.write("Traceback (most recent call last):\n")
            f.write("  File ...\n")
            f.write("RuntimeError: CUDA out of memory\n")
        with open(os.path.join(log_dir, "1.out"), "w") as f:
            f.write("epoch 4 val_loss=0.31\n")

        self.fake_stats = {
            ("12345", 0): slurm.JobStats(
                state="COMPLETED", elapsed="00:01:30",
                max_rss="1024K", ave_rss="512K", req_mem="1G",
            ),
            ("12345", 1): slurm.JobStats(state="RUNNING", elapsed="00:00:42"),
            ("12345", 2): slurm.JobStats(state="OUT_OF_MEMORY", elapsed="00:00:05"),
        }

    def _args(self, **overrides):
        defaults = dict(workspace=self.tmp, lines=20, max_failed=20)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_snapshot_bundles_status_stats_metrics_and_failed_stderr(self):
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
             mock.patch("hyperherd.slurm.query_job_stats", return_value=self.fake_stats), \
             mock.patch("hyperherd.slurm.get_log_tail",
                        side_effect=lambda ws, idx: "epoch 4 val_loss=0.31" if idx == 1 else ""):
            _, payload = self._capture_json(cmd_snapshot, self._args())

        self.assertEqual(payload["sweep_name"], "t")
        self.assertEqual(payload["totals"]["total"], 3)
        self.assertEqual(payload["totals"]["completed"], 1)
        self.assertEqual(payload["totals"]["running"], 1)
        self.assertEqual(payload["totals"]["failed"], 1)

        by_idx = {t["index"]: t for t in payload["trials"]}
        # completed trial carries sacct + metrics
        self.assertEqual(by_idx[0]["slurm_state"], "COMPLETED")
        self.assertEqual(by_idx[0]["elapsed_seconds"], 90)
        self.assertEqual(by_idx[0]["metrics"], {"test_acc": 0.95})
        self.assertEqual(by_idx[0]["slurm_job_id"], "12345")
        # running trial carries last_log_line
        self.assertIn("val_loss", by_idx[1]["last_log_line"])
        # failed trial has its sacct row even with no logged metrics
        self.assertEqual(by_idx[2]["slurm_state"], "OUT_OF_MEMORY")
        self.assertEqual(by_idx[2]["metrics"], {})

        # failed_stderr keyed by index, with the stderr lines the agent needs
        self.assertEqual(len(payload["failed_stderr"]), 1)
        block = payload["failed_stderr"][0]
        self.assertEqual(block["index"], 2)
        self.assertIn("RuntimeError: CUDA out of memory", block["stderr_lines"])
        self.assertFalse(block["stderr_truncated"])

    def test_snapshot_truncates_long_stderr(self):
        # Create 30 lines of stderr; ask for last 10
        log_dir = manifest.logs_path(self.tmp)
        with open(os.path.join(log_dir, "2.err"), "w") as f:
            for i in range(30):
                f.write(f"line {i}\n")
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
             mock.patch("hyperherd.slurm.query_job_stats", return_value=self.fake_stats), \
             mock.patch("hyperherd.slurm.get_log_tail", return_value=""):
            _, payload = self._capture_json(cmd_snapshot, self._args(lines=10))
        block = payload["failed_stderr"][0]
        self.assertEqual(len(block["stderr_lines"]), 10)
        self.assertEqual(block["stderr_lines"][0], "line 20")
        self.assertTrue(block["stderr_truncated"])
