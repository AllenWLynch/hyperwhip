"""Tests for the deterministic chat-command handlers.

These are pure functions over a workspace; we mock the `herd` subprocess
calls and exercise output formatting directly.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hyperherd.monitor_agent import commands as cmd_mod


def _fake_completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return mock.Mock(stdout=stdout, stderr=stderr, returncode=returncode)


class TestStatus(unittest.TestCase):
    def test_renders_totals_and_table(self):
        snapshot_json = (
            '{"sweep_name": "mnist_sweep", '
            '"totals": {"total": 3, "running": 1, "completed": 2}, '
            '"trials": ['
            '  {"index": 0, "status": "completed", "elapsed": "00:01:30", '
            '   "experiment_name": "lr-0.001_opt-adam"},'
            '  {"index": 1, "status": "running", "elapsed": "00:00:42", '
            '   "experiment_name": "lr-0.01_opt-sgd"},'
            '  {"index": 2, "status": "completed", "elapsed": "00:01:25", '
            '   "experiment_name": "lr-0.1_opt-adam"}'
            ']}'
        )
        with mock.patch.object(cmd_mod.subprocess, "run",
                               return_value=_fake_completed(stdout=snapshot_json)):
            out = cmd_mod.cmd_status(Path("/tmp/anything"))
        self.assertIn("mnist_sweep", out)
        self.assertIn("1 running", out)
        self.assertIn("2 completed", out)
        self.assertIn("3 total", out)
        self.assertIn("idx", out)
        self.assertIn("lr-0.001_opt-adam", out)

    def test_handles_subprocess_failure(self):
        err = cmd_mod.subprocess.CalledProcessError(
            2, "snap", stderr="boom\n", output="",
        )
        with mock.patch.object(cmd_mod.subprocess, "run", side_effect=err):
            out = cmd_mod.cmd_status(Path("/tmp/anything"))
        self.assertIn("failed", out)
        self.assertIn("boom", out)

    def test_handles_empty_workspace(self):
        snapshot_json = '{"sweep_name": "fresh", "totals": {}, "trials": []}'
        with mock.patch.object(cmd_mod.subprocess, "run",
                               return_value=_fake_completed(stdout=snapshot_json)):
            out = cmd_mod.cmd_status(Path("/tmp/anything"))
        self.assertIn("fresh", out)
        self.assertIn("no trials yet", out)


class TestStop(unittest.TestCase):
    def test_stop_one_returns_confirmation(self):
        with mock.patch.object(cmd_mod.subprocess, "run",
                               return_value=_fake_completed(stdout="Cancelled job 12345_0\n")):
            out = cmd_mod.cmd_stop(Path("/tmp/anything"), index=3)
        self.assertIn("Stopped trial 3", out)
        self.assertIn("12345_0", out)

    def test_stop_all_returns_confirmation(self):
        with mock.patch.object(cmd_mod.subprocess, "run",
                               return_value=_fake_completed(stdout="Cancelled 4 trials\n")):
            out = cmd_mod.cmd_stop_all(Path("/tmp/anything"))
        self.assertIn("Stopped all live trials", out)
        self.assertIn("Cancelled 4", out)


class TestTail(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)
        (self.workspace / ".hyperherd" / "logs").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_returns_last_n_lines(self):
        log = self.workspace / ".hyperherd" / "logs" / "5.err"
        log.write_text("\n".join(f"line-{i}" for i in range(50)))

        out = cmd_mod.cmd_tail(self.workspace, index=5, lines=10)
        self.assertIn("5.err", out)
        self.assertIn("line-49", out)
        self.assertIn("line-40", out)
        self.assertNotIn("line-39", out)

    def test_missing_log_returns_friendly_msg(self):
        out = cmd_mod.cmd_tail(self.workspace, index=99, lines=20)
        self.assertIn("No stderr log", out)
        self.assertIn("99", out)

    def test_empty_log_says_so(self):
        log = self.workspace / ".hyperherd" / "logs" / "7.err"
        log.write_text("")
        out = cmd_mod.cmd_tail(self.workspace, index=7, lines=20)
        self.assertIn("empty", out)

    def test_validates_lines_bounds(self):
        out = cmd_mod.cmd_tail(self.workspace, index=0, lines=0)
        self.assertIn("must be between", out)
        out = cmd_mod.cmd_tail(self.workspace, index=0, lines=10000)
        self.assertIn("must be between", out)


class TestHelp(unittest.TestCase):
    def test_lists_each_command(self):
        text = cmd_mod.cmd_help()
        for keyword in ("/status", "/stop", "/stop_all", "/tail", "/help",
                        "@HerdDog"):
            self.assertIn(keyword, text)


if __name__ == "__main__":
    unittest.main()
