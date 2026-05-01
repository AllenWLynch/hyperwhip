"""Tests for `herd stats` (sacct accounting per trial)."""

import argparse
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest, slurm
from hyperherd.cli import cmd_stats


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


class TestCmdStats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.01}],
            abbrevs={"lr": "lr"},
        )
        manifest.record_job_submission(self.tmp, "12345", [0, 1])
        self.fake_stats = {
            ("12345", 0): slurm.JobStats(
                state="COMPLETED", elapsed="00:01:30", max_rss="1234K",
                ave_rss="800K", req_mem="1G",
            ),
            ("12345", 1): slurm.JobStats(
                state="RUNNING", elapsed="00:00:42",
            ),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _args(self, **kw):
        base = dict(workspace=self.tmp, index=None, all=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_all_prints_table(self):
        with mock.patch(
            "hyperherd.cli.slurm.query_job_stats", return_value=self.fake_stats
        ):
            rc = cmd_stats(self._args(all=True))
        self.assertEqual(rc, 0)

    def test_single_index(self):
        with mock.patch(
            "hyperherd.cli.slurm.query_job_stats", return_value=self.fake_stats
        ):
            rc = cmd_stats(self._args(index=0))
        self.assertEqual(rc, 0)

    def test_unknown_index(self):
        with mock.patch(
            "hyperherd.cli.slurm.query_job_stats", return_value=self.fake_stats
        ):
            rc = cmd_stats(self._args(index=99))
        self.assertEqual(rc, 1)

    def test_neither_index_nor_all(self):
        rc = cmd_stats(self._args())
        self.assertEqual(rc, 1)

    def test_both_index_and_all(self):
        rc = cmd_stats(self._args(index=0, all=True))
        self.assertEqual(rc, 1)

    def test_no_jobs_recorded(self):
        # Workspace exists but no SLURM submission has been made yet.
        # Wipe the job_ids file and re-test.
        os.remove(manifest.job_ids_path(self.tmp))
        rc = cmd_stats(self._args(all=True))
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
