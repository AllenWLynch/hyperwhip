"""Tests for the init scaffolding."""

import os
import shutil
import stat
import tempfile
import unittest

from hyperherd.init import scaffold


class TestScaffold(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_files(self):
        config_path, launcher_path = scaffold(self.tmpdir)
        self.assertTrue(os.path.isfile(config_path))
        self.assertTrue(os.path.isfile(launcher_path))
        self.assertTrue(config_path.endswith("hyperherd.yaml"))
        self.assertTrue(launcher_path.endswith("launch.sh"))

    def test_name_derived_from_directory(self):
        subdir = os.path.join(self.tmpdir, "cool_experiment")
        os.makedirs(subdir)
        scaffold(subdir)
        with open(os.path.join(subdir, "hyperherd.yaml")) as f:
            content = f.read()
        self.assertIn("name: cool_experiment", content)

    def test_launcher_is_executable(self):
        _, launcher_path = scaffold(self.tmpdir)
        mode = os.stat(launcher_path).st_mode
        self.assertTrue(mode & stat.S_IEXEC)

    def test_launcher_contains_default_command(self):
        scaffold(self.tmpdir)
        with open(os.path.join(self.tmpdir, "launch.sh")) as f:
            content = f.read()
        self.assertIn("python train.py $OVERRIDES", content)

    def test_refuses_overwrite_by_default(self):
        scaffold(self.tmpdir)
        with self.assertRaises(FileExistsError):
            scaffold(self.tmpdir)

    def test_force_overwrites(self):
        scaffold(self.tmpdir)
        scaffold(self.tmpdir, overwrite=True)  # no error
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "hyperherd.yaml")))

    def test_from_config_copies_verbatim(self):
        src = os.path.join(self.tmpdir, "src.yaml")
        with open(src, "w") as f:
            f.write("name: from_source\nparameters:\n  foo:\n    type: discrete\n    values: [1]\n")
        dest = os.path.join(self.tmpdir, "dest")
        scaffold(dest, from_config=src)
        with open(os.path.join(dest, "hyperherd.yaml")) as f:
            content = f.read()
        self.assertIn("name: from_source", content)

    def test_from_launcher_copies_verbatim(self):
        src = os.path.join(self.tmpdir, "src_launch.sh")
        with open(src, "w") as f:
            f.write("#!/bin/bash\necho custom-launcher $1\n")
        dest = os.path.join(self.tmpdir, "dest")
        scaffold(dest, from_launcher=src)
        with open(os.path.join(dest, "launch.sh")) as f:
            content = f.read()
        self.assertIn("custom-launcher", content)


if __name__ == "__main__":
    unittest.main()
