"""Tests for SLURM utilities (no actual SLURM interaction)."""

import os
import shutil
import tempfile
import unittest

from hyperherd import manifest
from hyperherd.config import Config
from hyperherd.constraints import Trial
from hyperherd.slurm import _indices_to_array_spec, generate_sbatch_script


class TestIndicesToArraySpec(unittest.TestCase):
    def test_contiguous(self):
        self.assertEqual(_indices_to_array_spec([0, 1, 2, 3]), "0-3")

    def test_single(self):
        self.assertEqual(_indices_to_array_spec([5]), "5")

    def test_gaps(self):
        self.assertEqual(_indices_to_array_spec([0, 1, 3, 5, 6, 7]), "0-1,3,5-7")

    def test_all_individual(self):
        self.assertEqual(_indices_to_array_spec([1, 3, 5]), "1,3,5")

    def test_unsorted(self):
        self.assertEqual(_indices_to_array_spec([5, 3, 1, 2]), "1-3,5")

    def test_duplicates(self):
        self.assertEqual(_indices_to_array_spec([1, 1, 2, 2, 3]), "1-3")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _indices_to_array_spec([])


class _SbatchFixture:
    """Build a minimal real workspace + manifest for sbatch-script tests.

    The lookup-baking code needs an on-disk manifest to read trial data
    from, so each test class that exercises `generate_sbatch_script` builds
    one in a temp directory.
    """

    def __init__(self, max_concurrent=None, static_overrides=None):
        self.tmpdir = tempfile.mkdtemp()
        launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(launcher, 0o755)

        raw = {
            "name": "t",
            "workspace": self.tmpdir,
            "launcher": launcher,
            "parameters": {
                "lr": {"type": "discrete", "abbrev": "lr", "values": [0.1, 0.01]},
            },
            "grid": "all",
            "slurm": {
                "partition": "p", "time": "00:10:00",
                "mem": "1G", "cpus_per_task": 1,
            },
        }
        if max_concurrent is not None:
            raw["slurm"]["max_concurrent"] = max_concurrent
        if static_overrides is not None:
            raw["static_overrides"] = static_overrides
        self.cfg = Config.model_validate(raw)

        manifest.init_workspace(self.tmpdir)
        # Seed enough trials that any test index in 0..4 exists.
        combos = [
            Trial(params={"lr": v}) for v in [0.1, 0.01, 0.001, 0.0001, 0.00001]
        ]
        manifest.create_manifest(
            self.tmpdir, combos, self.cfg.abbrevs, self.cfg.labels
        )

    def cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class _SbatchTestCase(unittest.TestCase):
    """Mixin: tear down per-test workspaces created via _SbatchFixture."""

    def setUp(self):
        self._fixtures = []

    def tearDown(self):
        for f in self._fixtures:
            f.cleanup()

    def _fixture(self, **kwargs):
        f = _SbatchFixture(**kwargs)
        self._fixtures.append(f)
        return f


class TestArrayThrottle(_SbatchTestCase):
    def test_no_throttle(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 1, 2])
        self.assertIn("#SBATCH --array=0-2\n", script)

    def test_config_throttle(self):
        f = self._fixture(max_concurrent=4)
        script = generate_sbatch_script(f.cfg, [0, 1, 2, 3, 4])
        self.assertIn("#SBATCH --array=0-4%4\n", script)

    def test_cli_overrides_config(self):
        f = self._fixture(max_concurrent=4)
        script = generate_sbatch_script(f.cfg, [0, 1, 2, 3, 4], max_concurrent=2)
        self.assertIn("#SBATCH --array=0-4%2\n", script)

    def test_cli_only(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 1, 2], max_concurrent=1)
        self.assertIn("#SBATCH --array=0-2%1\n", script)


class TestSbatchLogAppend(_SbatchTestCase):
    def test_open_mode_append_set(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 1, 2])
        self.assertIn("#SBATCH --open-mode=append", script)

    def test_run_divider_emitted(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 1, 2])
        # Divider must reference the SLURM job + array task IDs and a timestamp.
        self.assertIn("HyperHerd run", script)
        self.assertIn("${SLURM_JOB_ID}", script)
        self.assertIn("${SLURM_ARRAY_TASK_ID}", script)
        self.assertIn("date -Iseconds", script)


class TestBakedLookup(_SbatchTestCase):
    """The sbatch script bakes per-trial overrides; compute nodes don't need Python."""

    def test_no_python_invocation_in_script(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 1, 2])
        # Must not call hyperherd-on-the-compute-node — that was the whole point.
        self.assertNotIn("python -m hyperherd", script)
        self.assertNotIn("python3 -m hyperherd", script)

    def test_case_block_has_entry_per_index(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 2, 4])
        self.assertIn('case "$SLURM_ARRAY_TASK_ID" in', script)
        # Each requested index gets its own case arm.
        self.assertIn("\n  0)\n", script)
        self.assertIn("\n  2)\n", script)
        self.assertIn("\n  4)\n", script)
        # Indices NOT in the submission list must not appear as arms.
        self.assertNotIn("\n  1)\n", script)
        self.assertNotIn("\n  3)\n", script)

    def test_case_block_sets_expected_vars(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0])
        self.assertIn("HYPERHERD_TRIAL_NAME=", script)
        # Legacy alias must still be set so existing trainer code keeps working.
        self.assertIn("HYPERHERD_EXPERIMENT_NAME=", script)
        self.assertIn("OVERRIDES=", script)
        self.assertIn("export HYPERHERD_TRIAL_NAME HYPERHERD_EXPERIMENT_NAME", script)

    def test_sweep_name_exported_at_top(self):
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0])
        # SWEEP_NAME is shared across all trials, so it goes at the top of
        # the script (alongside WORKSPACE/TRIAL_ID), not in the case block.
        self.assertIn(f"export HYPERHERD_SWEEP_NAME={f.cfg.name}", script)
        # And it must appear *before* the per-trial case block.
        self.assertLess(
            script.index("HYPERHERD_SWEEP_NAME"),
            script.index('case "$SLURM_ARRAY_TASK_ID"'),
        )

    def test_unknown_array_id_exits_nonzero(self):
        # The wildcard arm must fail loudly so a misconfigured array doesn't
        # silently run with empty overrides.
        f = self._fixture()
        script = generate_sbatch_script(f.cfg, [0, 1])
        self.assertIn("  *)", script)
        self.assertIn("exit 1", script)

    def test_static_overrides_baked_in(self):
        f = self._fixture(static_overrides=["data.path=/scratch", "max_epochs=10"])
        script = generate_sbatch_script(f.cfg, [0])
        self.assertIn("data.path=/scratch", script)
        self.assertIn("max_epochs=10", script)

    def test_static_override_with_special_chars_quoted_safely(self):
        # `shlex.quote` must defang spaces, dollars, and embedded quotes so the
        # bash `OVERRIDES=...` line stays syntactically valid.
        awkward = "data.path=/scratch with space/$USER/foo"
        f = self._fixture(static_overrides=[awkward])
        script = generate_sbatch_script(f.cfg, [0])
        # The whole OVERRIDES assignment is a single-quoted token, so the
        # `$USER` and embedded space don't get expanded by the shell.
        self.assertIn(awkward, script)
        self.assertIn("OVERRIDES='", script)

    def test_unknown_index_raises(self):
        f = self._fixture()
        # The fixture seeds 5 trials (0..4); submitting index 99 is a bug.
        with self.assertRaises(ValueError):
            generate_sbatch_script(f.cfg, [99])


class TestQueryJobStats(unittest.TestCase):
    """Unit-test the sacct row fusion (parent + .batch step) without invoking sacct."""

    def test_fuse_parent_and_batch_rows(self):
        from unittest import mock as _mock
        from hyperherd import slurm as _slurm
        sacct_out = (
            "12345_0|COMPLETED|00:01:30||||\n"
            "12345_0.batch|COMPLETED||1234K|800K||2048K\n"
            "12345_1|RUNNING|00:00:42||||\n"
        )
        fake = _mock.Mock(returncode=0, stdout=sacct_out, stderr="")
        with _mock.patch("hyperherd.slurm.subprocess.run", return_value=fake):
            stats = _slurm.query_job_stats(["12345"])
        self.assertEqual(stats[("12345", 0)].state, "COMPLETED")
        self.assertEqual(stats[("12345", 0)].elapsed, "00:01:30")
        self.assertEqual(stats[("12345", 0)].max_rss, "1234K")
        self.assertEqual(stats[("12345", 0)].ave_rss, "800K")
        self.assertEqual(stats[("12345", 0)].max_vm, "2048K")
        self.assertEqual(stats[("12345", 1)].state, "RUNNING")
        self.assertEqual(stats[("12345", 1)].elapsed, "00:00:42")
        self.assertEqual(stats[("12345", 1)].max_rss, "")  # no .batch row yet

    def test_throttled_compact_range_does_not_clobber_per_task_rows(self):
        """sacct emits per-task rows AND a compact `[7-10%4]` row for throttled
        arrays; the compact-range PENDING must not overwrite a real RUNNING
        on a task that already has its own row."""
        from unittest import mock as _mock
        from hyperherd import slurm as _slurm
        sacct_out = (
            "12345_0|COMPLETED|00:00:30||||\n"
            "12345_6|RUNNING|00:01:00||||\n"
            "12345_7|RUNNING|00:00:15||||\n"
            "12345_[7-10%4]|PENDING|00:00:00||||\n"
        )
        fake = _mock.Mock(returncode=0, stdout=sacct_out, stderr="")
        with _mock.patch("hyperherd.slurm.subprocess.run", return_value=fake):
            stats = _slurm.query_job_stats(["12345"])
        self.assertEqual(stats[("12345", 0)].state, "COMPLETED")
        self.assertEqual(stats[("12345", 6)].state, "RUNNING")
        self.assertEqual(stats[("12345", 7)].state, "RUNNING")  # not clobbered
        self.assertEqual(stats[("12345", 8)].state, "PENDING")  # filled by compact range
        self.assertEqual(stats[("12345", 9)].state, "PENDING")
        self.assertEqual(stats[("12345", 10)].state, "PENDING")

    def test_cancelled_compact_range_with_throttle_suffix(self):
        """A fully-cancelled throttled array shows up only as a compact range."""
        from unittest import mock as _mock
        from hyperherd import slurm as _slurm
        sacct_out = "99999_[0-4%2]|CANCELLED by 1000|00:00:00||||\n"
        fake = _mock.Mock(returncode=0, stdout=sacct_out, stderr="")
        with _mock.patch("hyperherd.slurm.subprocess.run", return_value=fake):
            stats = _slurm.query_job_stats(["99999"])
        for idx in range(5):
            self.assertEqual(stats[("99999", idx)].state, "CANCELLED")


if __name__ == "__main__":
    unittest.main()
