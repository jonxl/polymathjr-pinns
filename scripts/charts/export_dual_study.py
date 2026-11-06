#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///
"""Export saved Julia PanelSets to a portable, dated HTML study and PNG/PDF figures.

uv run scripts/charts/export_dual_study.py --geometry-root results/geometry
No training or checkpoint reinterpretation is performed by this exporter.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
from string import Template

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinns-study-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pinns-study-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).with_name("ui")


def export_browser_assets(output):
    destination = output / "ui"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (UI_DIR / "study.css", UI_DIR / "dist/study.js"):
        if not source.is_file():
            raise FileNotFoundError(f"Missing browser asset {source}; run bun install --frozen-lockfile and bun run build.")
        shutil.copyfile(source, destination / source.name)

REPS = ("power_series", "eigenvalue")
LABELS = ("Power series", "Eigenvalue")
EXPERIMENTS = {
    "family_disks": "Train on small family patches; measure plane error, radial extrapolation, and reach beyond the patch.",
    "spiral_transfer": "Train on the two spiral patches; measure transfer to the other ODE families.",
    "pair_transfer": "Compare training on paired spiral patches with training on paired node patches.",
    "parabola_segment": "Train on a segment of the repeated-root parabola; evaluate continuation along it and across the plane.",
    "parabola_slide": "Move a training window along the parabola; measure inward and outward generalization reach.",
    "parabola_curves": "Compare solution-error curves along the parabola for different training-window locations.",
    "parabola_bias": "Measure the difference between right-endpoint and left-endpoint errors as the training window moves.",
    "axis_transfer": "Train on patches along one coordinate axis; measure transfer to the other axis and the plane.",
    "axis_sweep": "Sweep patch centers along both axes; compare errors and reach as the training location changes.",
}
SECTIONS = {
    "shared_extrapolate": ("Extrapolation", "Train within a region, then measure solution error farther from the training data."),
    "shared_genradius_disk": ("Disk generalization", "Train inside a disk and map solution error across the trace–determinant plane."),
    "shared_genradius_family": ("Family generalization", "Train on each ODE family, map its solution error, and measure the area below 10% relative error."),
    "shared_range": ("Transfer across distance", "Measure transfer to other ODE families as the evaluation distance increases."),
    "shared_showcase": ("Solution examples", "Compare predicted solutions on training, held-out, and out-of-range equations."),
    "shared_transfer": ("Family transfer", "Train on one ODE family and test on each of the six families."),
}


def read(path):
    return json.loads(path.read_text())


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")


def esc(value):
    return html.escape(str(value), quote=True)


def load_training_history(path):
    fields = ("iteration", "total", "pde", "bc", "supervised")
    with path.open() as stream:
        reader = csv.DictReader(stream)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing loss columns: {path}")
        rows = [[float(row[k]) for k in fields] for row in reader if row.get("iteration", "").strip()]
    if not rows:
        raise ValueError(f"Empty training history: {path}")
    history = dict(zip(fields, np.asarray(rows, dtype=float).T))
    iterations = history["iteration"]
    if not np.all(np.isfinite(iterations)) or np.any(np.diff(iterations) <= 0):
        raise ValueError(f"Iterations must be finite and strictly increasing: {path}")
    return history


def render_training_history(history, rep, title, destination):
    """One model, one axis: logged total and raw components, without smoothing."""
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    supervised = "Coefficient supervision (raw)" if rep == "power_series" else "Solution supervision (raw)"
    labels = {"total": "Total objective (weighted)", "pde": "PDE residual (raw)",
              "bc": "Initial conditions (diagnostic)", "supervised": supervised}
    colors = {"total": "#24313a", "pde": "#245b91", "bc": "#b45309", "supervised": "#15803d"}
    losses = np.concatenate([history[key] for key in labels])
    finite = losses[np.isfinite(losses)]
    positive = finite[finite > 0]
    if len(positive) and np.all(finite > 0):
        ax.set_yscale("log")
    elif len(positive):
        ax.set_yscale("symlog", linthresh=float(positive.min()))
    for key, label in labels.items():
        y = np.where(np.isfinite(history[key]), history[key], np.nan)
        ax.plot(history["iteration"], y, label=label, color=colors[key],
                linewidth=2 if key == "total" else 1.3,
                linestyle="--" if key == "bc" else "-", marker="o", markersize=2)
    ax.set(title=title, xlabel="Training iteration", ylabel="Logged loss")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    notes = []
    if len(history["iteration"]) == 1:
        notes.append("Only one logged sample; no convergence history")
    if np.any(~np.isfinite(losses)):
        notes.append(f"{int(np.sum(~np.isfinite(losses)))} nonfinite values shown as gaps")
    if notes:
        fig.supxlabel(" · ".join(notes), fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(destination.with_suffix("."+extension), dpi=150)
    plt.close(fig)


def paired_figure_row(key, label, figures):
    cells = []
    for rep, name in zip(REPS, LABELS):
        cells.append(figures.get(rep, f'<figure><h3>{name}</h3><p>No saved result for this model.</p></figure>'))
    return f'<div class="figure-pair" data-chart-group="{esc(key)}" data-chart-label="{esc(label)}">{"".join(cells)}</div>'


def training_section(roots, assets, output):
    entries, figures, seen = [], {}, set()
    for root in roots:
        for path in sorted(root.rglob("loss.csv"), key=lambda p: (str(p.parent.parent), REPS.index(p.parent.name) if p.parent.name in REPS else 0)):
            if path.resolve() in seen:
                continue
            rep = next((r for r in REPS if r in path.parts), None)
            if rep is None:
                continue
            seen.add(path.resolve())
            history = load_training_history(path)
            identity = path.parent.relative_to(root).as_posix()
            label = LABELS[REPS.index(rep)]
            model_parts = path.parent.relative_to(root).parts
            model_name = '/'.join(model_parts[-3:-1] if 'family' in model_parts else model_parts[-2:-1])
            if model_parts[-1] not in REPS and len(model_parts)>=3:
                model_name = '/'.join((model_parts[-3],model_parts[-1]))
            title = f"{label} · {model_name.replace('_', ' ')}"
            name = f"training-{len(entries)+1:02d}-{slug(identity)}"
            render_training_history(history, rep, title, assets/name)
            shutil.copyfile(path, assets/(name+".csv"))
            relative = (assets/name).relative_to(output).as_posix()
            entry = {"source": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "representation": rep, "samples": len(history["iteration"]),
                     "first_iteration": float(history["iteration"][0]), "last_iteration": float(history["iteration"][-1]),
                     "figure": relative+".png", "data": relative+".csv"}
            entries.append(entry)
            group_key = '/'.join('representation' if part == rep else part for part in path.parent.resolve().parts)
            group = figures.setdefault(group_key, {"label": f'{model_name.replace("_", " ")} · {identity.rsplit("/", 1)[0]}', "figures": {}})
            group["figures"][rep] = f'<figure><h3>{esc(title)}</h3><p class="facts">{entry["samples"]} logged samples · iterations {entry["first_iteration"]:g}–{entry["last_iteration"]:g}<br>Run: {esc(identity)}</p><a href="{relative}.png"><img loading="lazy" src="{relative}.png" alt="{esc(title)}: total, PDE, initial-condition and supervised losses over iterations"></a><figcaption><a href="{relative}.pdf">PDF</a> · <a href="{relative}.png">PNG</a> · <a href="{relative}.csv">Loss CSV</a></figcaption></figure>'
    if not entries:
        return '', entries
    section = '<section id="training-history"><h2>Training histories</h2><p>One graph per base model: the recorded objective and all three loss components over training iterations. Curves use their own model’s scale, with no smoothing.</p><p>The total is weighted; the raw components need not sum to it. Initial-condition loss is a diagnostic outside the shared training objective. Supervision measures coefficient error for power series and solution error for eigenvalue models.</p><p class="facts">Check the run identity before relating a history to an evaluation map. The last logged iteration is not a claim of convergence.</p>'
    return section+''.join(paired_figure_row(key, group['label'], group['figures']) for key, group in figures.items())+'</section>', entries


def pair_panels(left, right):
    """Match by identity, never by incidental JSON ordering; reject changed designs."""
    lm, rm = left["meta"], right["meta"]
    if (lm.get("representation"), rm.get("representation")) != REPS:
        raise ValueError("Expected power_series on the left and eigenvalue on the right")
    for key in set(lm) | set(rm):
        if key != "representation" and lm.get(key) != rm.get(key):
            raise ValueError(f"Different experiment metadata: {key}")
    maps = [{p["id"]: p for p in s["panels"]} for s in (left, right)]
    if any(len(m) != len(s["panels"]) for m, s in zip(maps, (left, right))):
        raise ValueError("Duplicate panel IDs")
    if maps[0].keys() != maps[1].keys():
        raise ValueError("Representations have different panel IDs")
    for key, p in maps[0].items():
        q = maps[1][key]
        for field in ("kind", "xlabel", "ylabel"):
            if p.get(field) != q.get(field):
                raise ValueError(f"Different {field} in {key}")
        for field in ("categories", "row_labels", "col_labels", "x_values", "y_values", "training_patch"):
            if p["data"].get(field) != q["data"].get(field):
                raise ValueError(f"Different coordinates in {key}: {field}")
        for field in ("log_color", "log_y", "log_x", "floor_val", "yflip", "horizontal_lines", "vertical_lines"):
            if p.get("opts", {}).get(field) != q.get("opts", {}).get(field):
                raise ValueError(f"Different scale convention in {key}: {field}")
        if p["kind"] in ("lines", "scatter", "grouped_bar"):
            if [s["name"] for s in p["data"]["series"]] != [s["name"] for s in q["data"]["series"]]:
                raise ValueError(f"Different series in {key}")
            if p["kind"] != "grouped_bar" and [s["x"] for s in p["data"]["series"]] != [s["x"] for s in q["data"]["series"]]:
                raise ValueError(f"Different sample points in {key}")
        yield p, q


def values(panel):
    data, opts = panel["data"], panel.get("opts", {})
    if panel["kind"] == "heatmap":
        a = np.asarray(data["matrix"], dtype=float)
        return np.log10(np.maximum(a, opts.get("floor_val", 1e-20))) if opts.get("log_color") else a
    key = "values" if panel["kind"] == "grouped_bar" else "y"
    return np.concatenate([np.asarray(s[key], dtype=float) for s in data["series"]])


RELATIVE_CONTOURS = 5
FAILED_POINT_CAP = 1e6  # Matches plot_nonfinite_cap in saved geometry panels.


def relative_levels(z, log_color):
    """Evenly spaced levels across one surface's own range, ignoring failed points."""
    cap = np.log10(FAILED_POINT_CAP) if log_color else FAILED_POINT_CAP
    finite = z[np.isfinite(z) & (z < cap)]
    if finite.size == 0 or finite.min() == finite.max():
        return []
    return np.linspace(finite.min(), finite.max(), RELATIVE_CONTOURS+2)[1:-1]


def is_training_diagnostic(panel):
    """Raw optimization components do not establish comparative accuracy."""
    text = " ".join(str(panel.get(key, "")) for key in ("id", "title", "ylabel")).lower()
    return any(word in text for word in ("loss", "residual", "supervised", "component")) or panel["id"] in ("transfer_pde", "transfer_bc", "transfer_sup")


def render_pair(pair, destination, labels=LABELS):
    for panel in pair:
        if panel["kind"] in ("lines", "scatter"):
            for series in panel["data"]["series"]:
                if len(series["x"]) != len(series["y"]):
                    raise ValueError(f"Saved series {series['name']!r} has {len(series['x'])} x coordinates but {len(series['y'])} y values; regenerate this source panel")
    fig, axes = plt.subplots(1, len(pair), figsize=(7.5*len(pair), 5.8), layout="constrained", squeeze=False)
    axes = axes.ravel()
    guides = [float(line["value"]) for p in pair for line in p.get("opts", {}).get("horizontal_lines", [])]
    all_values = np.concatenate([values(p).ravel() for p in pair]+[np.asarray(guides)])
    finite = all_values[np.isfinite(all_values)]
    if pair[0].get("opts", {}).get("log_y"):
        finite = finite[finite > 0]
    lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
    if lo == hi:
        lo, hi = (lo / 2, hi * 2) if lo > 0 else (-.5, .5)
    is_area = pair[0]["id"] in ("genradius_areas", "geometry-areas", "outside-patch-area")
    use_log = pair[0].get("opts", {}).get("log_y") and len(finite) > 0 and not is_area
    for ax, panel, label in zip(axes, pair, labels):
        data, opts, kind = panel["data"], panel.get("opts", {}), panel["kind"]
        ax.set_title(label, loc="left", weight="bold")
        ax.set_xlabel(panel.get("xlabel", ""))
        ax.set_ylabel(panel.get("ylabel", ""))
        if kind == "heatmap":
            z = values(panel)
            xs, ys = data.get("x_values"), data.get("y_values")
            # Older plane panels store their coordinates as numeric tick labels.
            if xs is None and ys is None and (panel.get("xlabel"), panel.get("ylabel")) == ("τ", "Δ"):
                xs = [float(x) for x in data["col_labels"]]
                ys = [float(y) for y in data["row_labels"]]
            numeric = xs is not None and ys is not None
            xs = xs if numeric else np.arange(z.shape[1])
            ys = ys if numeric else np.arange(z.shape[0])
            hm = ax.pcolormesh(xs, ys, np.ma.masked_invalid(z), cmap="viridis", vmin=lo, vmax=hi, shading="nearest", rasterized=True)
            fig.colorbar(hm, ax=ax, label=opts.get("colorbar_label", "log10 relative L2"))
            if numeric:
                ax.set_aspect("equal")
                t = np.linspace(min(xs), max(xs), 400)
                ax.plot(t, t*t/4, color="black", linewidth=.8, linestyle="--", label="Δ = τ²/4")
                ax.set_ylim(min(ys), max(ys))
                levels = relative_levels(z, opts.get("log_color")) if opts.get("contour_levels") else []
                if len(levels):
                    contours = ax.contour(xs, ys, z, levels=levels, colors="white", linewidths=.5, linestyles="solid", alpha=.7)
                    ax.clabel(contours, fmt=lambda c: f"{10**c if opts.get('log_color') else c:.3g}", fontsize=7)
                for level in opts.get("contour_levels") or []:
                    c = np.log10(max(level, opts.get("floor_val", 1e-20))) if opts.get("log_color") else level
                    if np.nanmin(z) < c < np.nanmax(z):
                        contours = ax.contour(xs, ys, z, levels=[c], colors="white", linewidths=1)
                        ax.clabel(contours, fmt={c: f"ε={level:g}"}, fontsize=8)
                patch = data.get("training_patch")
                if patch:
                    points = np.asarray(patch["points"])
                    ax.scatter(points[:, 0], points[:, 1], s=9, facecolors="white", edgecolors="black", linewidths=.3, label="training equations", zorder=4)
                    if patch["kind"] == "disk":
                        ax.add_patch(plt.Circle(patch["center"], patch["radius"], fill=False, edgecolor="#dc2626", linewidth=1.5, label="training disk", zorder=5))
                    elif patch["kind"] == "segment":
                        cx, cy = patch["center"]
                        ax.plot([cx, cx], [cy-patch["radius"], cy+patch["radius"]], color="#dc2626", linewidth=2, label="training segment", zorder=5)
                    ax.legend(fontsize=7, loc="lower left")
                    ax.set_xlim(min(xs), max(xs))
                    ax.set_ylim(min(ys), max(ys))
            else:
                ax.set_xticks(xs, data["col_labels"], rotation=35, ha="right", fontsize=8)
                ax.set_yticks(ys, data["row_labels"], fontsize=8)
                if opts.get("yflip", True):
                    ax.invert_yaxis()
                if opts.get("annotate"):
                    for j, row in enumerate(data["matrix"]):
                        for i, value in enumerate(row):
                            ax.text(i, j, f"{value:.2g}" if value is not None else "failed", ha="center", va="center", fontsize=7, color="white" if z[j, i] < (lo+hi)/2 else "black")
        elif kind in ("lines", "scatter", "grouped_bar"):
            for i, series in enumerate(data["series"]):
                if kind == "grouped_bar":
                    width = .8 / len(data["series"])
                    x = np.arange(len(data["categories"])) + (i-(len(data["series"])-1)/2)*width
                    bars = ax.bar(x, series["values"], width=width, label=series["name"])
                    if is_area or opts.get("show_values"):
                        ax.bar_label(bars, fmt="%.3g", padding=4, fontsize=9)
                        ax.scatter(x, series["values"], s=18, color="#245b91", zorder=3)
                elif kind == "scatter":
                    ax.scatter(series["x"], series["y"], s=8, label=series["name"])
                else:
                    ax.plot(series["x"], series["y"], linewidth=1.5, label=series["name"])
                    if data.get("censored") is not None:
                        mask = np.asarray(data["censored"], dtype=bool)
                        ax.scatter(np.asarray(series["x"])[mask], np.asarray(series["y"])[mask], marker="^", color="#b45309", s=24, label="accurate at search cap")
            if kind == "grouped_bar":
                ax.set_xticks(np.arange(len(data["categories"])), data["categories"], rotation=35, ha="right", fontsize=8)
            if use_log:
                ax.set_yscale("log")
                ax.set_ylim(max(lo/1.2, np.finfo(float).tiny), hi*1.2)
            else:
                lower = min(0., lo) if kind == "grouped_bar" else lo
                pad = max((hi-lower)*.05, 1e-9)
                ax.set_ylim(lower-pad, hi+pad)
                if is_area:
                    ax.set_ylim(0, max(1., hi*1.2))
                    if not np.any(np.asarray(data["series"][0]["values"]) > 0):
                        ax.text(.5, .65, "No sampled point meets the error threshold", transform=ax.transAxes, ha="center", fontsize=10)
                elif opts.get("log_y"):
                    ax.text(.02, .97, "No positive values; linear scale shown", transform=ax.transAxes, va="top", fontsize=8)
            if opts.get("log_x"):
                ax.set_xscale("log")
            for line in opts.get("horizontal_lines", []):
                ax.axhline(line["value"], color="#b45309", linestyle="--", linewidth=1, label=line["label"])
            for line in opts.get("vertical_lines", []):
                ax.axvline(line["value"], color="#475569", linestyle=":", linewidth=1.5, label=line["label"])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="best")
        else:
            raise ValueError(f"Unsupported panel kind: {kind}")
    for extension in ("png", "pdf"):
        fig.savefig(destination.with_suffix("."+extension), dpi=150)
    plt.close(fig)


def caption(panel):
    note = ""
    if panel["kind"] == "heatmap" and panel.get("opts", {}).get("contour_levels"):
        note = " Thin lines step evenly across each panel's own error range, so compare panels by their labels."
    if panel.get("opts", {}).get("caption"):
        return panel["opts"]["caption"]+note
    identity = panel["id"]
    if identity == "genradius_areas":
        return "Estimated area with relative solution error below ε = 0.1 (10%). Zero means no sampled point meets this threshold; it does not mean missing data."
    if "sup" in identity or "component" in identity:
        return "Training diagnostics. Supervision is in coefficient space for power series and solution space for eigenvalue models; these component magnitudes are not equivalent accuracy metrics."
    if panel["kind"] == "heatmap":
        return "Both panels share one color scale. Contours appear only where their threshold crosses the saved data. Grid resolution is the saved evaluation resolution."+note
    return "Matching measurements with shared vertical limits. Raw saved values are preserved; changing the display does not retrain either model."




def build(args):
    date = dt.date.fromisoformat(args.date).isoformat()
    stem = f"dual-representation-{date}"
    output = args.output.resolve()
    assets = output / "figures" / stem
    assets.mkdir(parents=True, exist_ok=True)
    pairs = []
    for directory in args.panel_data:
        for path in sorted(directory.glob("*_power_series.json")):
            other = path.with_name(path.name.replace("_power_series.json", "_eigenvalue.json"))
            if other.exists():
                pairs.append((path.stem.removesuffix("_power_series"), path, other, False))
    for directory in args.geometry_root:
        for path in sorted(directory.rglob("power_series/panelset.json")):
            other = path.parent.parent / "eigenvalue" / "panelset.json"
            if other.exists():
                pairs.append((path.parent.parent.name, path, other, True))
    manifest = {"date": date, "source_commit": "bb2a102a", "comparisons": [], "geometry_status": {}}
    sections, counts, seen = [], {}, set()
    diagnostics = {}
    for number, (name, left_path, right_path, geometry) in enumerate(pairs, 1):
        identity = (left_path.resolve(), right_path.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        left, right = read(left_path), read(right_path)
        panels = list(pair_panels(left, right))
        tag = f"{number:02d}-{slug(name)}"
        meta = left["meta"]
        if geometry and meta.get("mode") == "matched_training" and not meta.get("design_sha256"):
            raise ValueError(f"Missing matched-training design fingerprint: {left_path}")
        experiment = meta.get("experiment")
        mode = meta.get("mode", "archived results; matching controls not recorded")
        if geometry:
            counts.setdefault(experiment, []).append(mode)
        run_group = left_path.parent.parent.parent.name if geometry else "Earlier saved experiments"
        entry = {"name": name, "run_group": run_group, "mode": mode, "metadata": meta, "sources": [], "figures": [], "training_diagnostics": [], "unavailable_figures": []}
        source_links = []
        for rep, path in zip(REPS, (left_path, right_path)):
            target = assets / f"{tag}-{rep}.json"
            shutil.copyfile(path, target)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entry["sources"].append({"path": str(path.resolve()), "sha256": digest})
            source_links.append(f'<a href="{target.relative_to(output).as_posix()}">{rep} data</a>')
        section_title, description = SECTIONS.get(name, (name.replace("_", " ").capitalize(), EXPERIMENTS.get(experiment, "Compare both representations on the same experiment.")))
        body = [f'<section id="{tag}" data-run-group="{esc(run_group)}"><h2>{esc(section_title)}</h2>', f'<p>{esc(description)}</p>']
        if "training_controls" in meta:
            controls = meta["training_controls"]
            body.append(f'<p class="facts">N = {int(controls["power_series_degree"])} · {int(controls["adam_iterations"]):,} Adam iterations · analytic evaluation on [{controls["x_left"]:g}, {controls["x_right"]:g}]</p>')
        if meta.get("fixture"):
            body.append(f'<p class="status">{esc(meta["fixture"])}</p>')
        if "Ng" in meta:
            body.append(f'<p class="facts">Evaluation grid: {int(meta["Ng"])} × {int(meta["Ng"])} equations. Grid density is separate from training quality.</p>')
        body.append(f'<details><summary>Data and settings</summary><p>{esc(mode)} · {" · ".join(source_links)}</p><pre>{esc(json.dumps(meta, indent=2))}</pre></details>')
        for index, pair in enumerate(panels, 1):
            filename = f"{tag}-{index:02d}-{slug(pair[0]['id'])}"
            if any(is_training_diagnostic(p) for p in pair):
                for rep, label, panel in zip(REPS, LABELS, pair):
                    diagnostic_name = f"{filename}-{rep}-diagnostic"
                    render_pair((panel,), assets / diagnostic_name, labels=(label+" · training diagnostic",))
                    relative = (assets / diagnostic_name).relative_to(output).as_posix()
                    title = panel["title"].replace(f" ({rep})", "")
                    group = diagnostics.setdefault(filename, {"label": f"{section_title} · {title}", "figures": {}})
                    group["figures"][rep] = f'<figure><h3>{esc(label)} · {esc(title)}</h3><a href="{relative}.png"><img loading="lazy" src="{relative}.png" alt="{esc(label)} training diagnostic: {esc(title)}"></a><figcaption>Within-model diagnostic; independently scaled. <a href="{relative}.pdf">PDF</a> · <a href="{relative}.png">PNG</a></figcaption></figure>'
                    entry["training_diagnostics"].append({"representation": rep, "figure": relative+".png", "scale": "independent"})
                continue
            try:
                render_pair(pair, assets / filename)
            except ValueError as error:
                entry["unavailable_figures"].append({"id": pair[0]["id"], "reason": str(error)})
                body.append(f'<div class="status"><h3>{esc(pair[0]["title"])}</h3><p>Figure unavailable: {esc(error)}.</p></div>')
                continue
            relative = (assets / filename).relative_to(output).as_posix()
            title = pair[0]["title"].replace(" (power_series)", "")
            body.append(f'<figure data-chart-id="{esc(pair[0]["id"])}"><h3>{esc(title)}</h3><a href="{relative}.png"><img loading="lazy" src="{relative}.png" alt="{esc(title)}: power series left, eigenvalue right"></a><figcaption>{esc(caption(pair[0]))} <a href="{relative}.pdf">PDF</a> · <a href="{relative}.png">PNG</a></figcaption></figure>')
            entry["figures"].append(relative+".png")
        body.append("</section>")
        sections.append((tag, section_title, "".join(body)))
        manifest["comparisons"].append(entry)
    rows = []
    for name, description in EXPERIMENTS.items():
        modes = counts.get(name, [])
        status = "Matched results available" if "matched_training" in modes else "Frozen probes only" if modes else "Results not found"
        manifest["geometry_status"][name] = status
        rows.append(f'<tr><th scope="row">{esc(name.replace("_", " ").capitalize())}</th><td>{esc(description)}</td></tr>')
    total = sum(len(c["figures"]) for c in manifest["comparisons"])
    completed = sum(s == "Matched results available" for s in manifest["geometry_status"].values())
    manifest_path = assets / "manifest.json"
    histories, history_entries = training_section(args.training_root + args.geometry_root, assets, output)
    manifest["training_histories"] = history_entries
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    nav = ' · '.join(f'<a href="#{tag}">{esc(name)}</a>' for tag, name, _ in sections)
    command = f"bash scripts/shared/run_geometry_study.sh {date}"
    diagnostic_section = ''
    if any(diagnostics.values()):
        diagnostic_section = '<section id="training-diagnostics"><h2>Loss diagnostics on evaluation equations</h2><p>These component measurements describe held-out equations, not training progress. Smaller raw losses do not establish better solution accuracy across representations.</p>'
        for key, group in diagnostics.items():
            diagnostic_section += paired_figure_row(key, group['label'], group['figures'])
        diagnostic_section += '</section>'
    export_browser_assets(output)
    page = Template((UI_DIR / "study.html").read_text()).substitute(
        title_date=date,
        date=date,
        figure_count=total,
        section_links=nav,
        training_link=' · <a href="#training-history">Training histories</a>' if histories else '',
        diagnostics_link=' · <a href="#training-diagnostics">Loss diagnostics</a>' if diagnostic_section else '',
        experiment_rows=''.join(rows),
        completed_count=completed,
        earlier_note=' The figures below also include earlier saved experiments.' if any((not pair[3] for pair in pairs)) else '',
        run_command=esc(command),
        sections=''.join((section for _, _, section in sections)),
        histories=histories,
        diagnostics=diagnostic_section,
        manifest_url=manifest_path.relative_to(output).as_posix(),
    )
    target = output / f"{stem}.html"
    target.write_text(page)
    print(f"{target}\n{total} paired figures; {completed}/9 matched geometry experiments")
    return target, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--output", type=Path, default=ROOT / "studies/webui")
    parser.add_argument("--panel-data", action="append", type=Path, help="Saved *_power_series.json / *_eigenvalue.json directory; repeatable")
    parser.add_argument("--geometry-root", action="append", type=Path, default=[], help="Recursively discover paired geometry panelsets; repeatable")
    parser.add_argument("--training-root", action="append", type=Path, help="Discover base-model loss.csv histories; repeatable")
    args = parser.parse_args()
    args.panel_data = args.panel_data if args.panel_data is not None else []
    args.training_root = args.training_root if args.training_root is not None else [ROOT / "results/dual-rep-studies"]
    if not args.geometry_root:
        args.geometry_root = [ROOT / "results/geometry"]
    build(args)


if __name__ == "__main__":
    main()
