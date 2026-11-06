"""Run with a Python environment containing matplotlib and numpy."""
import copy
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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
    def test_export_ships_external_browser_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            target, manifest = exporter.build(SimpleNamespace(
                date="2026-11-03", output=Path(directory),
                panel_data=[], geometry_root=[], training_root=[],
            ))
            page = target.read_text()
            self.assertNotIn("<style>", page)
            self.assertNotIn("<script>", page)
            self.assertNotIn("$date", page)
            self.assertIn('href="ui/study.css"', page)
            self.assertIn('src="ui/study.js"', page)
            self.assertEqual((target.parent / "ui/study.css").read_bytes(),
                             (exporter.UI_DIR / "study.css").read_bytes())
            self.assertEqual((target.parent / "ui/study.js").read_bytes(),
                             (exporter.UI_DIR / "dist/study.js").read_bytes())


    def test_disk_overlay_and_radial_guides_are_rendered(self):
        p = {"id": "saddle-plane", "title": "Saddle", "kind": "heatmap", "xlabel": "τ", "ylabel": "Δ",
             "opts": {"log_color": True, "floor_val": 1e-6, "contour_levels": [.1]},
             "data": {"matrix": [[.01,.2],[.02,.3]], "x_values": [-1.,1.], "y_values": [-1.,1.],
                      "training_patch": {"kind": "disk", "center": [0.,0.], "radius": .3, "points": [[0.,0.],[.1,.1]]}}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(exporter.plt, "close") as close:
                exporter.render_pair((p, copy.deepcopy(p)), Path(directory)/"disk")
                fig = close.call_args.args[0]
                self.assertEqual([circle.get_radius() for circle in fig.axes[0].patches], [.3])
                self.assertIn("training equations", fig.axes[0].get_legend_handles_labels()[1])
            exporter.plt.close(fig)
            curve = panel("shells")
            curve["opts"] = {"log_y": True, "horizontal_lines": [{"value": .1,"label": "tolerance"}],
                             "vertical_lines": [{"value": .3,"label": "patch edge"}]}
            with patch.object(exporter.plt, "close") as close:
                exporter.render_pair((curve, copy.deepcopy(curve)), Path(directory)/"shells")
                fig = close.call_args.args[0]
                self.assertIn("tolerance", fig.axes[0].get_legend_handles_labels()[1])
                self.assertIn("patch edge", fig.axes[0].get_legend_handles_labels()[1])
                self.assertLess(fig.axes[0].get_ylim()[0], .1)
            exporter.plt.close(fig)

    def test_relative_contours_follow_each_surface_range(self):
        z = exporter.np.log10(exporter.np.array([[.01, 10.], [1., exporter.FAILED_POINT_CAP]]))
        levels = 10**exporter.relative_levels(z, True)
        self.assertEqual(len(levels), exporter.RELATIVE_CONTOURS)
        self.assertTrue(((levels > .01) & (levels < 10.)).all())
        self.assertAlmostEqual(levels[1], .1)
        self.assertEqual(len(exporter.relative_levels(exporter.np.zeros((2, 2)), True)), 0)

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
