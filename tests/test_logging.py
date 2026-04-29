"""Tests for result logging."""

import json
import os
import shutil
import tempfile
import unittest

from hyperherd import manifest
from hyperherd.logging import log_result, load_trial_results, load_all_results


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


if __name__ == "__main__":
    unittest.main()
