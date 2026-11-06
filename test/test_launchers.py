"""Check launcher argument forwarding without starting Julia."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class LauncherTests(unittest.TestCase):
    def launch(self, script, args=(), config=None):
        with tempfile.TemporaryDirectory() as tmp:
            mock = Path(tmp)/"julia"
            mock.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            mock.chmod(0o755)
            env = dict(os.environ, JULIA_BIN=str(mock))
            env.pop("STUDY_CONFIG", None)
            if config:
                env["STUDY_CONFIG"] = config
            result = subprocess.run(["bash", str(ROOT/"scripts/shared"/script), *args], env=env, text=True, capture_output=True, check=True)
            return json.loads(result.stdout)

    def test_default_budget_and_dry_run(self):
        args = self.launch("run_generalization_study.sh", ["--dry-run"])
        self.assertEqual(args[args.index("--maxiters")+1], "1000000")
        self.assertIn("--dry-run", args)
        self.assertNotIn("--output-root", args)

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
