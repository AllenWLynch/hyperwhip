"""Tests for terminal output formatting."""

import unittest

from hyperherd.display import _condense_case_block, format_short_value


class TestCondenseCaseBlock(unittest.TestCase):
    """The dry-run printer trims the baked lookup case block for readability."""

    def _script(self, n_trials: int) -> str:
        arms = []
        for i in range(n_trials):
            arms.append(f"  {i})")
            arms.append(f"    HYPERHERD_TRIAL_NAME=trial-{i}")
            arms.append(f"    OVERRIDES='lr={0.1 ** i}'")
            arms.append("    ;;")
        return "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --array=0-{}".format(n_trials - 1),
                "",
                'case "$SLURM_ARRAY_TASK_ID" in',
                *arms,
                "  *)",
                '    echo "no entry" >&2',
                "    exit 1",
                "    ;;",
                "esac",
                "bash launch.sh \"$OVERRIDES\"",
            ]
        )

    def test_short_block_unchanged(self):
        # 1-2 trials: condensing would actually be longer than the original.
        script = self._script(2)
        self.assertEqual(_condense_case_block(script), script)

    def test_long_block_elided(self):
        script = self._script(20)
        out = _condense_case_block(script)
        # First arm preserved.
        self.assertIn("  0)\n", out)
        self.assertIn("HYPERHERD_TRIAL_NAME=trial-0", out)
        # Middle arms gone.
        self.assertNotIn("HYPERHERD_TRIAL_NAME=trial-10", out)
        self.assertNotIn("HYPERHERD_TRIAL_NAME=trial-19", out)
        # Wildcard arm preserved (so the user sees the safety net).
        self.assertIn("  *)", out)
        self.assertIn("exit 1", out)
        # Elision marker is visible and counts what's hidden.
        self.assertIn("19 more trial arm(s) elided", out)

    def test_no_case_block_passthrough(self):
        # A script without a case block (shouldn't happen in practice, but
        # the condenser must not corrupt it).
        plain = "#!/bin/bash\necho hello\n"
        self.assertEqual(_condense_case_block(plain), plain)


class TestFormatShortValue(unittest.TestCase):
    def test_float_uses_4g(self):
        self.assertEqual(format_short_value(0.0001234), "0.0001234")
        self.assertEqual(format_short_value(1234567.89), "1.235e+06")

    def test_int_unchanged(self):
        self.assertEqual(format_short_value(42), "42")

    def test_string_unchanged(self):
        self.assertEqual(format_short_value("adam"), "adam")


if __name__ == "__main__":
    unittest.main()
