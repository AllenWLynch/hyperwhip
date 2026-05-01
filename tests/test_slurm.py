"""Tests for SLURM utilities (no actual SLURM interaction)."""

import unittest

from hyperherd.config import Config
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


def _make_config(max_concurrent=None):
    raw = {
        "name": "t",
        "workspace": "/tmp/ws",
        "launcher": "/tmp/ws/launch.sh",
        "parameters": {"lr": {"type": "discrete", "abbrev": "lr", "values": [0.1, 0.01]}},
        "grid": "all",
        "slurm": {"partition": "p", "time": "00:10:00", "mem": "1G", "cpus_per_task": 1},
    }
    if max_concurrent is not None:
        raw["slurm"]["max_concurrent"] = max_concurrent
    return Config.model_validate(raw)


class TestArrayThrottle(unittest.TestCase):
    def test_no_throttle(self):
        cfg = _make_config()
        script = generate_sbatch_script(cfg, [0, 1, 2])
        self.assertIn("#SBATCH --array=0-2\n", script)

    def test_config_throttle(self):
        cfg = _make_config(max_concurrent=4)
        script = generate_sbatch_script(cfg, [0, 1, 2, 3, 4])
        self.assertIn("#SBATCH --array=0-4%4\n", script)

    def test_cli_overrides_config(self):
        cfg = _make_config(max_concurrent=4)
        script = generate_sbatch_script(cfg, [0, 1, 2, 3, 4], max_concurrent=2)
        self.assertIn("#SBATCH --array=0-4%2\n", script)

    def test_cli_only(self):
        cfg = _make_config()
        script = generate_sbatch_script(cfg, [0, 1, 2], max_concurrent=1)
        self.assertIn("#SBATCH --array=0-2%1\n", script)


class TestSbatchLogAppend(unittest.TestCase):
    def test_open_mode_append_set(self):
        cfg = _make_config()
        script = generate_sbatch_script(cfg, [0, 1, 2])
        self.assertIn("#SBATCH --open-mode=append", script)

    def test_run_divider_emitted(self):
        cfg = _make_config()
        script = generate_sbatch_script(cfg, [0, 1, 2])
        # Divider must reference the SLURM job + array task IDs and a timestamp.
        self.assertIn("HyperHerd run", script)
        self.assertIn("${SLURM_JOB_ID}", script)
        self.assertIn("${SLURM_ARRAY_TASK_ID}", script)
        self.assertIn("date -Iseconds", script)


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


if __name__ == "__main__":
    unittest.main()
