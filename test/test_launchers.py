"""Check launcher argument forwarding without starting Julia."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class LauncherTests(unittest.TestCase):
    def launch(self, script, args=(), config=None, iterations=None, extra_env=None):
        with tempfile.TemporaryDirectory() as tmp:
            mock = Path(tmp)/"julia"
            mock.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            mock.chmod(0o755)
            env = dict(os.environ, JULIA_BIN=str(mock))
            env.pop("STUDY_CONFIG", None)
            env.pop("STUDY_ITERATIONS", None)
            for name in ("STUDY_LANES", "STUDY_DEGREES"):
                env.pop(name, None)
            env.update(extra_env or {})
            if iterations is not None:
                env["STUDY_ITERATIONS"] = str(iterations)
            if config:
                env["STUDY_CONFIG"] = config
            result = subprocess.run(["bash", str(ROOT/"scripts/shared"/script), *args], env=env, text=True, capture_output=True, check=True)
            return json.loads(result.stdout)

    def test_default_budget_and_dry_run(self):
        args = self.launch("run_generalization_study.sh", ["--dry-run"])
        self.assertEqual(args[args.index("--maxiters")+1], "10000")
        self.assertIn("--dry-run", args)
        self.assertNotIn("--output-root", args)

    def test_environment_budget_override(self):
        for budget in (19000, 1000000):
            args = self.launch("run_generalization_study.sh", iterations=budget,
                               config="configs/power_series_standard.toml")
            self.assertEqual(args[args.index("--maxiters")+1], str(budget))
        args = self.launch("run_generalization_study.sh", ["", "", "23"], iterations=19000)
        self.assertEqual(args[args.index("--maxiters")+1], "23")
        with self.assertRaises(subprocess.CalledProcessError):
            self.launch("run_generalization_study.sh", iterations="19K")

    def test_environment_lanes_and_degrees(self):
        args = self.launch("run_generalization_study.sh")
        for flag in ("--lanes", "--degrees"):
            self.assertNotIn(flag, args)
        args = self.launch("run_generalization_study.sh", extra_env={"STUDY_LANES": "8", "STUDY_DEGREES": "20,25,30,35"})
        self.assertEqual(args[args.index("--lanes")+1], "8")
        self.assertEqual(args[args.index("--degrees")+1], "20,25,30,35")
        for bad in ({"STUDY_LANES": "0"}, {"STUDY_DEGREES": "20,many"}, {"STUDY_DEGREES": "20,"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.launch("run_generalization_study.sh", extra_env=bad)

    def test_presets_keep_their_budget_unless_explicitly_overridden(self):
        for script in ("run_generalization_study.sh", "run_geometry_study.sh"):
            args = self.launch(script, config="configs/power_series_standard.toml")
            self.assertNotIn("--maxiters", args)
            self.assertIn("--config", args)
            args = self.launch(script, ["2026-11-06", "a path with spaces", "17"], config="configs/power_series_standard.toml")
            self.assertEqual(args[args.index("--maxiters")+1], "17")
            self.assertEqual(args[args.index("--output-root")+1], "a path with spaces")

if __name__ == "__main__":
    unittest.main()
