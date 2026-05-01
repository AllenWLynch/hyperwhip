"""Tests for result logging and override parsing."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.logging import (
    load_all_results,
    load_trial_results,
    log_result,
    parse_overrides,
)


class TestLogResult(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        # Set env vars as mush would
        os.environ["HYPERHERD_WORKSPACE"] = self.tmpdir
        os.environ["HYPERHERD_TRIAL_ID"] = "3"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)

    def test_log_single_metric(self):
        log_result("accuracy", 0.95)
        results = load_trial_results(self.tmpdir, 3)
        self.assertEqual(results["accuracy"], 0.95)

    def test_log_multiple_metrics(self):
        log_result("accuracy", 0.95)
        log_result("loss", 0.12)
        log_result("epochs", 50)
        results = load_trial_results(self.tmpdir, 3)
        self.assertEqual(results["accuracy"], 0.95)
        self.assertEqual(results["loss"], 0.12)
        self.assertEqual(results["epochs"], 50)

    def test_overwrite_metric(self):
        log_result("accuracy", 0.5)
        log_result("accuracy", 0.95)
        results = load_trial_results(self.tmpdir, 3)
        self.assertEqual(results["accuracy"], 0.95)

    def test_missing_env_vars(self):
        os.environ.pop("HYPERHERD_WORKSPACE")
        with self.assertRaises(RuntimeError):
            log_result("x", 1)

    def test_missing_trial_id(self):
        os.environ.pop("HYPERHERD_TRIAL_ID")
        with self.assertRaises(RuntimeError):
            log_result("x", 1)


class TestLoadAllResults(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        os.environ["HYPERHERD_WORKSPACE"] = self.tmpdir

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)

    def test_load_multiple_trials(self):
        os.environ["HYPERHERD_TRIAL_ID"] = "0"
        log_result("acc", 0.9)
        os.environ["HYPERHERD_TRIAL_ID"] = "1"
        log_result("acc", 0.8)
        os.environ["HYPERHERD_TRIAL_ID"] = "2"
        log_result("acc", 0.95)

        all_results = load_all_results(self.tmpdir)
        self.assertEqual(len(all_results), 3)
        self.assertEqual(all_results[0]["acc"], 0.9)
        self.assertEqual(all_results[2]["acc"], 0.95)

    def test_empty_results(self):
        all_results = load_all_results(self.tmpdir)
        self.assertEqual(all_results, {})


class TestParseOverrides(unittest.TestCase):
    """Round-trip values from the override-string format into Python types."""

    def test_basic_kv_pairs(self):
        out = parse_overrides("lr=0.001 batch_size=64 optimizer=adam")
        self.assertEqual(out, {"lr": 0.001, "batch_size": 64, "optimizer": "adam"})

    def test_bool_and_null(self):
        out = parse_overrides("use_amp=true verbose=false note=null other=None")
        self.assertEqual(
            out, {"use_amp": True, "verbose": False, "note": None, "other": None}
        )

    def test_float_with_exponent(self):
        out = parse_overrides("lr=1e-3 wd=2.5e-4")
        self.assertAlmostEqual(out["lr"], 0.001)
        self.assertAlmostEqual(out["wd"], 2.5e-4)

    def test_signed_int_and_float(self):
        out = parse_overrides("seed=-1 momentum=-0.9")
        self.assertEqual(out["seed"], -1)
        self.assertIsInstance(out["seed"], int)
        self.assertAlmostEqual(out["momentum"], -0.9)

    def test_string_value_unchanged(self):
        out = parse_overrides("optimizer=adam exp_name=lr-0.001_opt-adam")
        self.assertEqual(out["optimizer"], "adam")
        self.assertEqual(out["exp_name"], "lr-0.001_opt-adam")

    def test_skips_non_kv_tokens(self):
        # Tokens without `=` (e.g. trailing `--cfg job` from herd test --cfg-job)
        # are silently dropped — parser is meant to extract params, not flags.
        out = parse_overrides("lr=0.001 --cfg job optimizer=adam")
        self.assertEqual(out, {"lr": 0.001, "optimizer": "adam"})

    def test_empty_string(self):
        self.assertEqual(parse_overrides(""), {})

    def test_reads_sys_argv_when_omitted(self):
        with mock.patch.object(sys, "argv", ["train.py", "lr=0.5 epochs=3"]):
            out = parse_overrides()
        self.assertEqual(out, {"lr": 0.5, "epochs": 3})

    def test_raises_when_no_arg_and_argv_empty(self):
        with mock.patch.object(sys, "argv", ["train.py"]):
            with self.assertRaises(RuntimeError):
                parse_overrides()


if __name__ == "__main__":
    unittest.main()
