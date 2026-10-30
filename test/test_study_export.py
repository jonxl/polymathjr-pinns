"""Run with a Python environment containing matplotlib and numpy."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

sys.dont_write_bytecode = True

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/charts/export_dual_study.py"
spec = importlib.util.spec_from_file_location("study_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def panel(identity):
    return {"id": identity, "title": identity, "kind": "lines", "xlabel": "x", "ylabel": "error",
            "opts": {"log_y": False}, "data": {"series": [{"name": "error", "x": [0, 1], "y": [1, 2]}]}}


def sets():
    return [{"meta": {"representation": rep, "design_sha256": "same"},
             "panels": [panel("first"), panel("second")]} for rep in exporter.REPS]


class StudyExportTests(unittest.TestCase):
    def test_paired_layout_always_places_power_series_first(self):
        row = exporter.paired_figure_row("model", "Model", {
            "eigenvalue": "<figure>EV result</figure>",
            "power_series": "<figure>PS result</figure>",
        })
        self.assertLess(row.index("PS result"), row.index("EV result"))
        self.assertIn('class="figure-pair"', row)
        missing = exporter.paired_figure_row("model", "Model", {"eigenvalue": "<figure>EV result</figure>"})
        self.assertLess(missing.index("No saved result"), missing.index("EV result"))

    def test_history_preserves_actual_iterations_and_failed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"loss.csv"
            path.write_text("iteration,total,bc,pde,supervised\n1,10,0,2,3\n100,NaN,0,1,2\n250,4,0,.5,1\n")
            h = exporter.load_training_history(path)
            self.assertEqual(list(h["iteration"]), [1, 100, 250])
            self.assertTrue(exporter.np.isnan(h["total"][1]))
            target = Path(directory)/"history"
            exporter.render_training_history(h, "power_series", "Example model", target)
            self.assertGreater(target.with_suffix(".png").stat().st_size, 1000)
            path.write_text("iteration,total,bc,pde,supervised\n100,1,0,1,0\n1,1,0,1,0\n")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                exporter.load_training_history(path)

    def test_loss_diagnostics_are_separate_from_solution_metrics(self):
        for identity, title in (("transfer_sup", "Supervised component"),
                                ("transfer_bc", "IC component"),
                                ("transfer_pde", "PDE residual"),
                                ("component_bars", "Loss components"),
                                ("training_total", "Total loss")):
            p = panel(identity)
            p["title"] = title
            self.assertTrue(exporter.is_training_diagnostic(p))
        for identity in ("transfer_relerr", "genradius_areas", "showcase_bars", "reach-margins"):
            self.assertFalse(exporter.is_training_diagnostic(panel(identity)))

    def test_diagnostic_can_render_one_representation_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"diagnostic"
            exporter.render_pair((panel("transfer_sup"),), target, labels=("Power series",))
            self.assertGreater(target.with_suffix(".png").stat().st_size, 1000)

    def test_pair_by_id_despite_reordered_json(self):
        left, right = sets()
        right["panels"].reverse()
        pairs = list(exporter.pair_panels(left, right))
        self.assertEqual([(a["id"], b["id"]) for a, b in pairs], [("first", "first"), ("second", "second")])

    def test_rejects_mismatched_controls_and_coordinates(self):
        left, right = sets()
        right["meta"]["design_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "metadata"):
            list(exporter.pair_panels(left, right))
        right["meta"]["design_sha256"] = "same"
        right["panels"][0]["data"]["series"][0]["x"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "sample points"):
            list(exporter.pair_panels(left, right))

    def test_malformed_saved_curve_is_not_interpolated_or_fabricated(self):
        p = panel("bad")
        p["data"]["series"][0]["y"] = [1]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "2 x coordinates but 1 y"):
                exporter.render_pair((p, copy.deepcopy(p)), Path(directory)/"bad")
            self.assertFalse(list(Path(directory).iterdir()))

    def test_renders_downloadable_figures_with_zero_log_data(self):
        p = panel("zero")
        p["opts"]["log_y"] = True
        p["data"]["series"][0]["y"] = [0, 0]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"zero"
            exporter.render_pair((p, copy.deepcopy(p)), target)
            self.assertGreater(target.with_suffix(".png").stat().st_size, 1000)
            self.assertTrue(target.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
