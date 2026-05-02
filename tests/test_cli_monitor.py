"""Tests for `herd monitor` (the agent launcher).

The handler ends with `os.execvp("claude", ...)`, which would replace the
test process if we let it run. All tests mock execvp so we can capture the
call and assert side effects (background watch spawn, prompt file written)
without actually launching Claude Code.
"""

import argparse
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import _MONITOR_ALLOW_RULES, cmd_monitor


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


class TestCmdMonitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp, [{"lr": 0.1}, {"lr": 0.01}], abbrevs={"lr": "lr"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _args(self, **overrides):
        defaults = dict(
            workspace=self.tmp, no_watch=True, stop=False,
            no_auto_allow=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_execs_claude_with_prompt_as_argv(self):
        # `claude "<prompt>"` (positional, no -p) starts an interactive session
        # with the prompt already running — that's how we avoid making the user
        # paste anything.
        with mock.patch("os.execvp") as exec_mock:
            cmd_monitor(self._args())

        exec_mock.assert_called_once()
        prog, argv = exec_mock.call_args.args
        self.assertEqual(prog, "claude")
        self.assertEqual(argv[0], "claude")
        self.assertEqual(len(argv), 2)
        injected_prompt = argv[1]
        self.assertIn(os.path.abspath(self.tmp), injected_prompt)
        self.assertIn("hyperherd-monitor", injected_prompt)
        # Dynamic /loop (no interval) so the agent self-paces.
        self.assertIn("/loop", injected_prompt)
        self.assertIn("ScheduleWakeup", injected_prompt)

        # The audit-trail file is still written so the user can see exactly
        # what the agent was told on this run.
        prompt_path = os.path.join(self.tmp, ".hyperherd", "monitor-prompt.txt")
        self.assertTrue(os.path.isfile(prompt_path))
        self.assertEqual(open(prompt_path).read().rstrip("\n"), injected_prompt)

    def test_no_watch_skips_background_spawn(self):
        with mock.patch("os.execvp"), \
             mock.patch("hyperherd.cli._spawn_watch_background") as spawn_mock:
            cmd_monitor(self._args(no_watch=True))
        spawn_mock.assert_not_called()

    def test_default_spawns_background_watch(self):
        with mock.patch("os.execvp"), \
             mock.patch("hyperherd.cli._spawn_watch_background",
                        return_value=(99999, "/tmp/x.pid", "/tmp/x.log")) as spawn_mock:
            cmd_monitor(self._args(no_watch=False))
        spawn_mock.assert_called_once_with(self.tmp)

    def test_stop_kills_pidfile_pid(self):
        # Plant a pidfile pointing at our own PID (which exists), then assert
        # cmd_monitor --stop sends a signal and reports success.
        ws_dir = manifest.workspace_path(self.tmp)
        pidfile = os.path.join(ws_dir, "watch.pid")
        with open(pidfile, "w") as f:
            f.write(f"{os.getpid()}\n")
        with mock.patch("os.kill") as kill_mock:
            rc = cmd_monitor(self._args(stop=True))
        self.assertEqual(rc, 0)
        kill_mock.assert_called_once_with(os.getpid(), 15)

    def test_stop_with_no_pidfile_is_no_op(self):
        # Pidfile doesn't exist — should report cleanly and exit 0.
        rc = cmd_monitor(self._args(stop=True))
        self.assertEqual(rc, 0)

    def test_claude_missing_returns_error(self):
        with mock.patch("os.execvp", side_effect=FileNotFoundError):
            rc = cmd_monitor(self._args())
        self.assertEqual(rc, 1)

    def test_no_workspace_refuses(self):
        # A bare directory with the YAML but no `.hyperherd/` should fail
        # before we reach exec — there's nothing to monitor.
        empty = tempfile.mkdtemp()
        try:
            _write_config(empty)
            args = argparse.Namespace(
                workspace=empty, no_watch=True, stop=False,
                no_auto_allow=False,
            )
            with mock.patch("os.execvp") as exec_mock:
                rc = cmd_monitor(args)
            self.assertEqual(rc, 1)
            exec_mock.assert_not_called()
        finally:
            shutil.rmtree(empty)


class TestMonitorAutoAllow(unittest.TestCase):
    """The agent runs unattended; Claude Code's default ask-before-each-call
    behavior would defeat that. `herd monitor` writes a project-scope
    `.claude/settings.local.json` with an allowlist on first run."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp, [{"lr": 0.1}, {"lr": 0.01}], abbrevs={"lr": "lr"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _args(self, **overrides):
        defaults = dict(
            workspace=self.tmp, no_watch=True, stop=False,
            no_auto_allow=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _read_settings(self) -> dict:
        path = os.path.join(self.tmp, ".claude", "settings.local.json")
        with open(path) as f:
            return json.load(f)

    def test_auto_allow_writes_settings_file(self):
        with mock.patch("os.execvp"):
            cmd_monitor(self._args())
        data = self._read_settings()
        for rule in _MONITOR_ALLOW_RULES:
            self.assertIn(rule, data["permissions"]["allow"])
        # Regressions: these specific entries previously bit users in the
        # field — agents pipeline through `tee` and reach for `grep`/`head`,
        # and missing them caused a tick to stall on a permission prompt.
        for rule in ("Bash(tee *)", "Bash(grep *)", "Bash(head *)"):
            self.assertIn(rule, data["permissions"]["allow"])

    def test_auto_allow_merges_with_existing_settings(self):
        # Plant a workspace settings file with an unrelated allow entry.
        settings_dir = os.path.join(self.tmp, ".claude")
        os.makedirs(settings_dir, exist_ok=True)
        with open(os.path.join(settings_dir, "settings.local.json"), "w") as f:
            json.dump(
                {
                    "permissions": {"allow": ["Bash(git status)"]},
                    "model": "claude-sonnet-4-6",
                },
                f,
            )
        with mock.patch("os.execvp"):
            cmd_monitor(self._args())
        data = self._read_settings()
        # Pre-existing entries preserved
        self.assertIn("Bash(git status)", data["permissions"]["allow"])
        self.assertEqual(data.get("model"), "claude-sonnet-4-6")
        # New entries added
        for rule in _MONITOR_ALLOW_RULES:
            self.assertIn(rule, data["permissions"]["allow"])

    def test_auto_allow_dedupes(self):
        # Run twice; the allowlist shouldn't double up.
        with mock.patch("os.execvp"):
            cmd_monitor(self._args())
        with mock.patch("os.execvp"):
            cmd_monitor(self._args())
        data = self._read_settings()
        for rule in _MONITOR_ALLOW_RULES:
            self.assertEqual(data["permissions"]["allow"].count(rule), 1)

    def test_no_auto_allow_skips_write(self):
        with mock.patch("os.execvp"):
            cmd_monitor(self._args(no_auto_allow=True))
        path = os.path.join(self.tmp, ".claude", "settings.local.json")
        self.assertFalse(os.path.exists(path))
