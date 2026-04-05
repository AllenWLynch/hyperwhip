"""Tests for preflight validation checks."""

import os
import shutil
import stat
import tempfile
import unittest

from hyperwhip.config import (
    Config, Constraint, HydraConfig, ParameterSpec, SearchConfig, SlurmConfig,
)
from hyperwhip.preflight import PreflightError, run_preflight


def _make_config(
    parameters=None,
    constraints=None,
    launcher=None,
    workspace=None,
    search_mode="grid",
    defaults=None,
):
    """Build a Config for testing. Creates a real launcher file if launcher is None."""
    if parameters is None:
        parameters = [ParameterSpec(name="lr", abbrev="lr", type="discrete", values=[0.1, 0.01])]
    if constraints is None:
        constraints = []

    return Config(
        name="test",
        workspace=workspace or "/tmp/hyperwhip_preflight_test",
        search=SearchConfig(mode=search_mode, defaults=defaults),
        slurm=SlurmConfig(),
        hydra=HydraConfig(),
        launcher=launcher or "",
        parameters=parameters,
        constraints=constraints,
    )


class TestLauncherCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_missing_launcher(self):
        config = _make_config(launcher="")
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("No launcher script", str(ctx.exception))

    def test_nonexistent_launcher(self):
        config = _make_config(launcher="/tmp/nonexistent_launcher_xyz.sh")
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("not found", str(ctx.exception))

    def test_non_executable_launcher(self):
        launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(launcher, stat.S_IRUSR | stat.S_IWUSR)  # rw, not x

        config = _make_config(launcher=launcher, workspace=self.tmpdir)
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("not executable", str(ctx.exception))

    def test_valid_launcher(self):
        launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(launcher, stat.S_IRWXU)

        config = _make_config(launcher=launcher, workspace=self.tmpdir)
        # Should not raise
        run_preflight(config)


class TestWorkspaceCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a valid launcher for all tests in this class
        self.launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(self.launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(self.launcher, stat.S_IRWXU)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_nonexistent_parent(self):
        config = _make_config(
            launcher=self.launcher,
            workspace="/tmp/no_such_parent_xyz/no_such_child/ws",
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("parent directory", str(ctx.exception))

    def test_existing_writable_workspace(self):
        ws = os.path.join(self.tmpdir, "ws")
        os.makedirs(ws)
        config = _make_config(launcher=self.launcher, workspace=ws)
        run_preflight(config)  # should not raise


class TestParameterChecks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(self.launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(self.launcher, stat.S_IRWXU)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_empty_discrete_values(self):
        params = [ParameterSpec(name="x", abbrev="x", type="discrete", values=[])]
        config = _make_config(launcher=self.launcher, workspace=self.tmpdir, parameters=params)
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("empty values", str(ctx.exception))

    def test_continuous_low_ge_high(self):
        params = [ParameterSpec(name="x", abbrev="x", type="continuous", low=1.0, high=0.5, steps=3)]
        config = _make_config(launcher=self.launcher, workspace=self.tmpdir, parameters=params)
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("less than high", str(ctx.exception))

    def test_log_scale_negative_low(self):
        params = [ParameterSpec(name="x", abbrev="x", type="continuous", low=-1.0, high=1.0, scale="log", steps=3)]
        config = _make_config(launcher=self.launcher, workspace=self.tmpdir, parameters=params)
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("log scale requires low > 0", str(ctx.exception))


class TestConstraintRefChecks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(self.launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(self.launcher, stat.S_IRWXU)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_unknown_when_param(self):
        params = [ParameterSpec(name="lr", abbrev="lr", type="discrete", values=[0.1])]
        constraints = [Constraint(name="c1", when={"nonexistent": "val"}, exclude={"lr": [0.1]})]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, constraints=constraints,
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("unknown parameter 'nonexistent'", str(ctx.exception))

    def test_unknown_exclude_param(self):
        params = [ParameterSpec(name="lr", abbrev="lr", type="discrete", values=[0.1])]
        constraints = [Constraint(name="c1", when={"lr": 0.1}, exclude={"bogus": [1]})]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, constraints=constraints,
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("unknown parameter 'bogus'", str(ctx.exception))

    def test_unknown_force_param(self):
        params = [ParameterSpec(name="lr", abbrev="lr", type="discrete", values=[0.1])]
        constraints = [Constraint(name="c1", when={"lr": 0.1}, force={"missing": 99})]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, constraints=constraints,
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("unknown parameter 'missing'", str(ctx.exception))

    def test_valid_constraint_refs(self):
        params = [
            ParameterSpec(name="lr", abbrev="lr", type="discrete", values=[0.1, 0.01]),
            ParameterSpec(name="opt", abbrev="opt", type="discrete", values=["adam", "sgd"]),
        ]
        constraints = [Constraint(name="c1", when={"opt": "sgd"}, exclude={"lr": [0.1]})]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, constraints=constraints,
        )
        run_preflight(config)  # should not raise


class TestAxesDefaultsCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(self.launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(self.launcher, stat.S_IRWXU)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_axes_default_not_in_values(self):
        params = [ParameterSpec(name="opt", abbrev="opt", type="discrete", values=["adam", "sgd"])]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, search_mode="axes", defaults={"opt": "rmsprop"},
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("not in its values list", str(ctx.exception))

    def test_axes_continuous_default_out_of_range(self):
        params = [ParameterSpec(name="lr", abbrev="lr", type="continuous", low=0.001, high=0.1, steps=3)]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, search_mode="axes", defaults={"lr": 999.0},
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("outside range", str(ctx.exception))

    def test_axes_valid_defaults(self):
        params = [
            ParameterSpec(name="lr", abbrev="lr", type="continuous", low=0.001, high=0.1, steps=3),
            ParameterSpec(name="opt", abbrev="opt", type="discrete", values=["adam", "sgd"]),
        ]
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters=params, search_mode="axes", defaults={"lr": 0.01, "opt": "adam"},
        )
        run_preflight(config)  # should not raise


if __name__ == "__main__":
    unittest.main()
