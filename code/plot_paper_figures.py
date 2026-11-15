"""Build the predeclared quantitative supplemental figures.

All plotted values are read from the project result files. The script does not
generate, resample, or replace analysis results. Figure 1 and Figure 2 require
formal result files; the pilot artifacts are never opened by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import runpy
from datetime import datetime, timezone
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter


ROOT = source_root() / 'additional_analyses'
DATA_ROOT = source_root()
RESULTS = DATA_ROOT / "results"
FIGURES = run_root() / "figures"
QA = run_root() / "qa"
CONTRACT_FILE = run_root() / 'docs/figure_contracts.json'
CONTRACT_FILE.parent.mkdir(parents=True,exist_ok=True)
if not CONTRACT_FILE.exists():
    CONTRACT_FILE.write_bytes((source_root() / 'docs/figure_contracts.json').read_bytes())
MANIFEST_FILE = QA / "figure_manifest.json"

BLUE = "#0072B2"
ORANGE = "#D98200"
GRAY = "#747B80"
LIGHT_GRAY = "#B9C0C5"
DARK_GRAY = "#3F4B52"
PALE_BLUE = "#DCEAF2"
GRID = "#E6EAED"
DOUBLE_COLUMN_WIDTH_IN = 7.20  # 183 mm double-column figure width

FIXED_METHODS = [
    {
        "key": "legacy_covariance",
        "label": "Legacy covariance",
        "short": "Legacy",
        "color": GRAY,
        "aliases": ["delta_legacy_covariance", "legacy_cov", "legacy_covariance"],
        "variance": "variance_legacy",
        "variance_ok": "legacy_variance_ok",
    },
    {
        "key": "corrected_common_cr1_tw",
        "label": "Corrected common CR1 two-way",
        "short": "Common TW",
        "color": ORANGE,
        "aliases": [
            "delta_corrected_common_cr1_tw",
            "correctedcommonTW",
            "corrected_common_tw",
        ],
        "variance": "variance_common_cr1",
        "variance_ok": "common_cr1_variance_ok",
    },
    {
        "key": "corrected_component_cr1_tw",
        "label": "Corrected component-specific two-way",
        "short": "Component TW",
        "color": BLUE,
        "aliases": [
            "delta_corrected_cr1_tw",
            "correctedcomponentTW",
            "corrected_component_tw",
        ],
        "variance": "variance_corrected",
        "variance_ok": "corrected_variance_ok",
    },
    {
        "key": "independent_HC1",
        "label": "HC1",
        "short": "HC1",
        "color": LIGHT_GRAY,
        "aliases": [
            "delta_independent_HC1_positive_control",
            "HC1",
            "hc1",
        ],
        "variance": "variance_hc1",
        "variance_ok": "hc1_variance_ok",
    },
    {
        "key": "region_CR1",
        "label": "Region CR1",
        "short": "Region CR1",
        "color": DARK_GRAY,
        "aliases": ["delta_region_cr1_context", "regionCR1", "region_cr1"],
        "variance": "variance_region_cr1",
        "variance_ok": "region_cr1_variance_ok",
    },
]

REAL_METHODS = [
    {"key": "model", "label": "Poisson", "short": "Poisson", "color": GRAY, "marker": "o", "aliases": ["model", "poisson"]},
    {"key": "hac14", "label": "HAC14", "short": "HAC14", "color": BLUE, "marker": "s", "aliases": ["hac14"]},
    {"key": "hac28", "label": "HAC28", "short": "HAC28", "color": ORANGE, "marker": "^", "aliases": ["hac28"]},
    {
        "key": "quasi_poisson_pearson_scale",
        "label": "Quasi-Poisson",
        "short": "Quasi-Poisson",
        "color": DARK_GRAY,
        "marker": "D",
        "aliases": ["quasi_poisson_pearson_scale", "quasi_poisson", "quasi", "quasipoisson"],
    },
]

OBSERVED_REAL_METHODS = REAL_METHODS[:3]

SCENARIO_ORDER = [
    "model_compatible",
    "response_misspecification",
    "omitted_confounding",
    "smooth_omitted_interaction",
    "correlated_overdispersed_ar1",
]

HEATDAY_SCENARIO_ORDER = ["model_compatible", "source_h_quadratic", "common_bias_0_12"]
HEATDAY_SCENARIO_LABELS = {
    "model_compatible": "Model-compatible",
    "source_h_quadratic": "Heat-day quadratic",
    "common_bias_0_12": "Quadratic + shared\ntemporal term",
}


def set_publication_style() -> None:
    private_fonts = Path(
        os.environ.get("CA_FIGURE_FONT_DIR", Path(__file__).resolve().parents[1] / "fonts")
    ).expanduser()
    for font_name in ("arial.ttf", "arialbd.ttf"):
        font_path = private_fonts / font_name
        if font_path.is_file():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except (OSError, RuntimeError):
                pass
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8.2,
            "font.weight": "bold",
            "axes.labelsize": 8.2,
            "axes.labelweight": "bold",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "mathtext.default": "bf",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "axes.unicode_minus": False,
        }
    )


def apply_figure_typography(fig: plt.Figure) -> None:
    for artist in fig.findobj(match=Text):
        if artist.get_text().strip():
            artist.set_fontweight("bold")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_name(path: Path) -> str:
    resolved = path.resolve()
    for base, label in ((ROOT.resolve(), "revision"), (DATA_ROOT.resolve(), "frozen_source")):
        try:
            return f"{label}/{resolved.relative_to(base).as_posix()}"
        except ValueError:
            continue
    return path.name


def source_metadata(path: Path, rows: int | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {"file": relative_name(path), "sha256": sha256_file(path)}
    if rows is not None:
        output["rows"] = int(rows)
    return output


def write_source_bundle(
    stem: str,
    frame: pd.DataFrame,
    *,
    sources: list[Path],
    figure_legend: dict[str, Any],
) -> tuple[Path, Path]:
    QA.mkdir(parents=True, exist_ok=True)
    csv_path = QA / f"figure_sources_{stem}.csv"
    json_path = QA / f"figure_sources_{stem}.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.12g")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        parsed_rows = list(csv.reader(handle))
    data_rows = parsed_rows[1:]
    if len(data_rows) != len(frame):
        raise ValueError(f"Plotted source CSV row count mismatch for {stem}.")
    row_hashes = [
        hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        for row in data_rows
    ]
    payload = {
        "schema_version": 1,
        "figure": stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [source_metadata(path) for path in sources],
        "figure_legend": figure_legend,
        "plotted_data": frame.where(pd.notna(frame), None).to_dict(orient="records"),
        "csv_export": source_metadata(csv_path, rows=len(frame)),
        "source_row_hashes": {
            "algorithm": "SHA-256 over each parsed CSV row serialized as compact UTF-8 JSON, preserving field order",
            "rows": row_hashes,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    apply_figure_typography(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    svg_path = FIGURES / f"{stem}.svg"
    tiff_path = FIGURES / f"{stem}.tiff"
    # Keep the declared 183 mm canvas; do not let tight-bbox export change width.
    fig.savefig(png_path, dpi=300, bbox_inches=None)
    fig.savefig(pdf_path, bbox_inches=None)
    fig.savefig(svg_path, bbox_inches=None)
    fig.savefig(tiff_path, dpi=600, bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(tiff_path) as image:
        image.convert("RGB").save(tiff_path, compression="tiff_lzw",dpi=(600,600))
    plt.close(fig)
    return [pdf_path, png_path, svg_path, tiff_path]


def update_manifest(figure_name: str, outputs: list[Path], sources: list[Path]) -> None:
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "figures": {}}
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["runtime"] = {
        "python": sys.version.split()[0],
        "matplotlib": mpl.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "backend": mpl.get_backend(),
    }
    manifest["script"] = source_metadata(Path(__file__))
    manifest.setdefault("figures", {})[figure_name] = {
        "outputs": [source_metadata(path) for path in outputs],
        "sources": [source_metadata(path) for path in sources],
        "png_dpi": 300,
        "vector_text": "SVG text remains text; PDF uses TrueType font embedding (fonttype 42)."
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def update_contract_status(figure_key: str, status: str) -> None:
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    contract["figures"][figure_key]["status"] = status
    CONTRACT_FILE.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")


def build_monthly_figure() -> None:
    source = RESULTS / "monthly_bound_repair.json"
    if not source.exists():
        raise FileNotFoundError(f"Required result is missing: {relative_name(source)}")
    data = json.loads(source.read_text(encoding="utf-8"))
    expected_deltas = [float(value) for value in data["predeclared_sensitivity_grid"]]
    if expected_deltas != [0.01, 0.025, 0.05, 0.1]:
        raise ValueError(f"Unexpected predeclared delta grid: {expected_deltas}")
    if abs(float(data["main_delta"]) - 0.05) > 1e-12:
        raise ValueError("The monthly main delta must remain the predeclared 0.05.")

    labels = {
        "BRA": "Cholecystectomy admission authorisations",
        "MEX": "Cholecystectomy discharges",
        "SIM": "All-cause registered deaths",
        "MEXD": "All-cause registered deaths",
    }
    order = ["BRA", "MEX", "SIM", "MEXD"]
    rows: list[dict[str, Any]] = []
    for panel in order:
        panel_data = data["panels"][panel]
        if panel_data.get("status") != "ok":
            raise ValueError(f"Monthly panel {panel} is not marked ok.")
        sensitivity = panel_data["sensitivity_class"]
        panel_grid = sensitivity["delta_grid_results"]
        found_deltas = [float(result["delta"]) for result in panel_grid]
        if found_deltas != expected_deltas:
            raise ValueError(f"{panel} does not contain the full predeclared delta grid.")
        weights = sensitivity["allocation_weights"]
        if len(weights) != 2 or not np.allclose(weights, [0.5, 0.5], atol=1e-12):
            raise ValueError(f"Unexpected lag allocation for {panel}: {weights}")
        for result in panel_grid:
            lower, upper = map(float, result["interval_bounds"])
            sens_lower, sens_upper = map(float, result["A_sensitivity_bounds"])
            point = float(result["A_hat"])
            if not (lower <= point <= upper):
                raise ValueError(f"Expanded interval does not contain A_hat for {panel}.")
            rows.append(
                {
                    "panel": panel,
                    "outcome": labels[panel],
                    "retained_region_month_rows": int(panel_data["panel_qc"]["used_rows"]),
                    "regions": int(panel_data["panel_qc"]["regions"]),
                    "calendar_months": int(panel_data["panel_qc"]["calendar_months"]),
                    "delta": float(result["delta"]),
                    "main_delta": bool(abs(float(result["delta"]) - float(data["main_delta"])) < 1e-12),
                    "lag0_allocation_weight": float(weights[0]),
                    "lag1_allocation_weight": float(weights[1]),
                    "A_hat_expected_outcome_count": point,
                    "base_90pct_half_width": float(result["base_half_width"]),
                    "sensitivity_radius": float(result["sensitivity_radius"]),
                    "sensitivity_lower": sens_lower,
                    "sensitivity_upper": sens_upper,
                    "expanded_interval_lower": lower,
                    "expanded_interval_upper": upper,
                    "zero_exclusion_status": result["threshold_diagnostics"]["status"],
                    "class_id": sensitivity["class_id"],
                    "finite_grid_only": not bool(sensitivity["continuous_temperature_certificate"]),
                    "uniform_exponential_remainder": bool(sensitivity["uniform_exponential_remainder"]),
                }
            )
    source_frame = pd.DataFrame(rows)
    source_frame.to_csv(
        QA / "figure_sources_monthly.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    # Only the four predeclared grid values are shown. The line segments do not
    # represent a continuous-delta certificate; error bars are deterministic bounds.
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.95), sharex=True)
    axes_flat = axes.ravel()
    letters = ["A", "B", "C", "D"]
    for panel_index, (ax, panel, letter) in enumerate(zip(axes_flat, order, letters)):
        panel_rows = source_frame[source_frame["panel"] == panel].sort_values("delta")
        x = panel_rows["delta"].to_numpy(float)
        point = panel_rows["A_hat_expected_outcome_count"].to_numpy(float)
        lower = panel_rows["expanded_interval_lower"].to_numpy(float)
        upper = panel_rows["expanded_interval_upper"].to_numpy(float)
        colors = [ORANGE if is_main else BLUE for is_main in panel_rows["main_delta"]]
        for xi, yi, lo, hi, color in zip(x, point, lower, upper, colors):
            ax.errorbar(
                xi,
                yi,
                yerr=[[yi - lo], [hi - yi]],
                fmt="o",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                ecolor=color,
                elinewidth=1.25,
                capsize=3.0,
                capthick=1.0,
                zorder=3,
            )
        ax.axhline(0, color=GRAY, linestyle=(0, (3, 2)), linewidth=0.75, zorder=0)
        ax.axvline(float(data["main_delta"]), color=ORANGE, alpha=0.22, linewidth=0.9, zorder=0)
        panel_title = f"{panel}  |  {labels[panel]}"
        if panel == "BRA":
            panel_title = "BRA  |  Cholecystectomy admission\nauthorisations"
        ax.set_title(panel_title, loc="left", pad=4, fontweight="bold")
        ax.text(
            -0.115,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_xticks(expected_deltas, ["0.01", "0.025", "0.05", "0.10"])
        ax.set_xlim(0.002, 0.108)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.tick_params(axis="both", length=2.5, width=0.65, color=GRAY)
        ax.spines["left"].set_color(GRAY)
        ax.spines["bottom"].set_color(GRAY)
        ax.set_xlabel("Delta (log-risk perturbation bound)" if panel_index >= 2 else "")
        ax.set_ylabel("Expected count contrast\n(factual − 1°C cooler path)" if panel_index % 2 == 0 else "")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=5, label="Other predeclared delta values"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=5, label="Main delta = 0.05"),
        Line2D([0], [0], color=GRAY, linestyle=(0, (3, 2)), linewidth=0.8, label="Zero reference"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.2,
        handletextpad=0.45,
        borderaxespad=0,
    )
    fig.suptitle(
        "Monthly expected-count contrast sensitivity bounds",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.16, top=0.88, hspace=0.42, wspace=0.31)

    outputs = save_figure(fig, "Figure_6")
    legend = {
        "claim": "At the predeclared delta=0.05, only SIM's expanded interval for the expected count contrast (factual minus the 1°C cooler path) excludes zero, on the negative side; the other three panels include zero.",
        "mark": "Each point is the observed fitted A_hat; each vertical bar is the approximate nominal 90% model-based sampling interval expanded by the finite-grid sensitivity radius, conditional on the stated fixed-allocation class.",
        "n": "Retained region-month rows: BRA 3213; MEX 3808; SIM 1917; MEXD 3808. The observed input panel, not a resampled replicate, is the unit.",
        "uncertainty": "These are deterministic sensitivity bounds around an approximate model-based 90% interval, not empirical coverage intervals or Monte Carlo error bars.",
        "class": "Two-lag fixed allocation with weights 0.5/0.5; constraints cover the finite observed factual and counterfactual basis grid; no continuous-temperature certificate or automatic 1/0.85 correction.",
        "scale": "Separate y-axis ranges by panel; all values are expected count contrasts, defined as factual minus the 1°C cooler path. Cholecystectomy outcome labels refer to hospital admission/discharge records, and death outcomes are all-cause registered deaths.",
    }
    csv_path, json_path = write_source_bundle(
        "monthly",
        source_frame,
        sources=[source],
        figure_legend=legend,
    )
    update_manifest("figure_6_monthly_sensitivity", outputs, [source, csv_path, json_path])
    update_contract_status("figure_6_monthly_sensitivity", "rendered; visual QA pending")


def _normalize(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _method_for_name(value: Any, methods: list[dict[str, Any]]) -> dict[str, Any] | None:
    name = _normalize(value)
    for method in methods:
        aliases = {_normalize(alias) for alias in method.get("aliases", [method["key"]])}
        aliases.add(_normalize(method["key"]))
        if name in aliases:
            return method
    return None


def _first_present(row: pd.Series, names: list[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in row.index:
            return name
    if required:
        raise KeyError(f"None of the expected columns were present: {names}")
    return None


def _bool_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _read_fixed_panels() -> tuple[list[pd.DataFrame], list[Path]]:
    files = sorted(RESULTS.glob("fixed_formal_panel_*.csv"))
    if not files:
        raise FileNotFoundError("Formal fixed-design panel CSVs are required for Figure 1.")
    frames: list[pd.DataFrame] = []
    for path in files:
        if "pilot" in path.name.lower():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "mode" in frame:
            frame = frame[frame["mode"].astype(str).str.lower() == "formal"]
        if "stage" in frame:
            frame = frame[frame["stage"].astype(str).str.lower() == "test"]
        if len(frame):
            frame["_source_file"] = path.name
            frames.append(frame)
    if not frames:
        raise ValueError("No formal test-stage rows were found in fixed-design panel CSVs.")
    return frames, files


def _scenario_label(value: str) -> str:
    labels = {
        "model_compatible": "Model-compatible",
        "response_misspecification": "Response misspecification",
        "omitted_confounding": "Omitted confounding",
        "correlated_overdispersed_ar1": "Correlated overdispersed AR(1)",
        "smooth_omitted_interaction": "Omitted seasonal interaction",
    }
    return labels.get(value, value.replace("_", " ").title())


def build_fixed_figure() -> None:
    summary_path = RESULTS / "fixed_formal_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Formal fixed-design summary is required for Figure 1.")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    panel_frames, panel_paths = _read_fixed_panels()
    if "mode" in summary and not (summary["mode"].astype(str).str.lower() == "formal").all():
        summary = summary[summary["mode"].astype(str).str.lower() == "formal"]

    selected = summary.copy()
    selected["_method"] = selected["method"].map(lambda value: _method_for_name(value, FIXED_METHODS))
    selected = selected[selected["_method"].notna()].copy()
    selected["_method_key"] = selected["_method"].map(lambda method: method["key"])
    if selected.empty:
        raise ValueError("No selected covariance methods were found in the formal summary.")

    coverage_col = _first_present(selected.iloc[0], ["coverage", "coverage_planned", "coverage_planned_failures_uncovered"])
    mcse_col = _first_present(selected.iloc[0], ["coverage_mcse_1se", "coverage_mcse_planned", "coverage_mcse"])
    n_col = _first_present(selected.iloc[0], ["n_planned", "n_test"])
    fail_col = _first_present(selected.iloc[0], ["n_fit_fail", "n_failed", "fit_failures"], required=False)
    invalid_col = _first_present(selected.iloc[0], ["n_method_interval_invalid_after_fit", "n_interval_invalid_after_fit"], required=False)

    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(selected["scenario"].astype(str))]
    if len(scenarios) != 3:
        raise ValueError(f"Expected three formal scenarios, found: {scenarios}")
    seeds = sorted(pd.to_numeric(selected["design_seed"], errors="raise").astype(int).unique().tolist())
    if len(seeds) != 3:
        raise ValueError(f"Expected three formal design seeds, found: {seeds}")

    rows: list[dict[str, Any]] = []
    for _, summary_row in selected.iterrows():
        scenario = str(summary_row["scenario"])
        seed = int(summary_row["design_seed"])
        method = summary_row["_method"]
        matches: list[pd.DataFrame] = []
        for panel in panel_frames:
            if "scenario" not in panel or "design_seed" not in panel:
                continue
            mask = (panel["scenario"].astype(str) == scenario) & (
                pd.to_numeric(panel["design_seed"], errors="coerce") == seed
            )
            if len(panel[mask]):
                matches.append(panel[mask])
        if not matches:
            raise ValueError(f"No formal test rows for {scenario}, seed {seed}.")
        replications = pd.concat(matches, ignore_index=True)
        if len(replications) != int(float(summary_row[n_col])):
            raise ValueError(
                f"Panel rows ({len(replications)}) do not match planned n ({summary_row[n_col]}) for {scenario}/{seed}."
            )
        if int(float(summary_row[n_col])) != 1000:
            raise ValueError(f"Figure 1 expects 1,000 planned test repetitions, got {summary_row[n_col]}.")

        fit_col = _first_present(replications.iloc[0], ["coefficient_fit_ok", "fit_ok"])
        variance_col = method["variance"]
        if variance_col not in replications:
            raise KeyError(f"Missing replicate variance column {variance_col}.")
        fit_mask = _bool_values(replications[fit_col])
        variance = pd.to_numeric(replications[variance_col], errors="coerce")
        valid_mask = fit_mask & np.isfinite(variance) & (variance > 0)
        if method["variance_ok"] in replications:
            valid_mask &= _bool_values(replications[method["variance_ok"]])
        se_values = np.sqrt(variance[valid_mask].to_numpy(float))

        if "error" in replications:
            error_values = pd.to_numeric(replications.loc[fit_mask, "error"], errors="coerce").to_numpy(float)
            error_values = error_values[np.isfinite(error_values)]
        else:
            error_values = (
                pd.to_numeric(replications.loc[fit_mask, "A_hat"], errors="coerce")
                - pd.to_numeric(replications.loc[fit_mask, "A_true"], errors="coerce")
            ).to_numpy(float)
            error_values = error_values[np.isfinite(error_values)]
        if not len(error_values):
            raise ValueError(f"No fitted test errors for {scenario}/{seed}/{method['key']}.")
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": _scenario_label(scenario),
                "design_seed": seed,
                "method": method["key"],
                "method_label": method["label"],
                "coverage": float(summary_row[coverage_col]),
                "coverage_mcse_1se": float(summary_row[mcse_col]),
                "n_planned": int(float(summary_row[n_col])),
                "fit_failures_counted_uncovered": int(float(summary_row[fail_col])) if fail_col and pd.notna(summary_row[fail_col]) else int((~fit_mask).sum()),
                "invalid_method_intervals_after_fit": int(float(summary_row[invalid_col])) if invalid_col and pd.notna(summary_row[invalid_col]) else int((fit_mask & ~valid_mask).sum()),
                "mean_standard_error": float(np.mean(se_values)) if len(se_values) else np.nan,
                "n_mean_standard_error": int(len(se_values)),
                "empirical_error_sd": float(np.std(error_values, ddof=1)) if len(error_values) > 1 else np.nan,
                "n_empirical_error_sd": int(len(error_values)),
                "coefficient_fit_successes": int(fit_mask.sum()),
                "panel_test_rows": int(len(replications)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["scenario", "design_seed", "method"]).any():
        raise ValueError("Duplicate formal summary records for scenario, design seed and method.")
    if len(frame) != 3 * 3 * 5:
        raise ValueError(f"Expected 45 plotted fixed-design summary points, got {len(frame)}.")
    frame.to_csv(QA / "figure_sources_fixed.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.65), sharey="row")
    method_positions = np.arange(len(FIXED_METHODS), dtype=float)
    seed_offsets = np.linspace(-0.16, 0.16, len(seeds))
    for column, scenario in enumerate(scenarios):
        scenario_data = frame[frame["scenario"] == scenario]
        ax_cov = axes[0, column]
        for m_index, method in enumerate(FIXED_METHODS):
            subset = scenario_data[scenario_data["method"] == method["key"]].sort_values("design_seed")
            for seed_index, (_, point) in enumerate(subset.iterrows()):
                ax_cov.errorbar(
                    method_positions[m_index] + seed_offsets[seed_index],
                    point["coverage"],
                    yerr=point["coverage_mcse_1se"],
                    fmt="o",
                    markersize=3.5,
                    color=method["color"],
                    ecolor=method["color"],
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.9,
                    zorder=3,
                )
        ax_cov.axhline(0.90, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
        ax_cov.set_ylim(0.0, 1.01)
        ax_cov.set_xlim(-0.5, len(FIXED_METHODS) - 0.5)
        ax_cov.set_xticks(method_positions, [method["short"] for method in FIXED_METHODS], rotation=25, ha="right")
        ax_cov.set_ylabel("Coverage" if column == 0 else "")
        ax_cov.set_title(_scenario_label(scenario), loc="left", pad=4, fontweight="bold")
        ax_cov.text(-0.13, 1.04, chr(ord("A") + column), transform=ax_cov.transAxes, fontsize=18, fontweight="bold")
        ax_cov.grid(axis="y", color=GRID, linewidth=0.5)
        ax_cov.tick_params(axis="both", length=2.3, width=0.6, color=GRAY)

        ax_scale = axes[1, column]
        for m_index, method in enumerate(FIXED_METHODS):
            subset = scenario_data[scenario_data["method"] == method["key"]].sort_values("design_seed")
            for seed_index, (_, point) in enumerate(subset.iterrows()):
                xpos = method_positions[m_index] + seed_offsets[seed_index]
                ax_scale.plot(
                    xpos - 0.025,
                    point["mean_standard_error"],
                    marker="o",
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgecolor=method["color"],
                    markeredgewidth=0.9,
                    linestyle="none",
                    zorder=3,
                )
                ax_scale.plot(
                    xpos + 0.025,
                    point["empirical_error_sd"],
                    marker="s",
                    markersize=2.8,
                    markerfacecolor=method["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    linestyle="none",
                    zorder=3,
                )
        ax_scale.set_xlim(-0.5, len(FIXED_METHODS) - 0.5)
        ax_scale.set_xticks(method_positions, [method["short"] for method in FIXED_METHODS], rotation=25, ha="right")
        ax_scale.set_ylabel("Error scale\n(expected counts)" if column == 0 else "")
        ax_scale.grid(axis="y", color=GRID, linewidth=0.5)
        ax_scale.tick_params(axis="both", length=2.3, width=0.6, color=GRAY)
        ax_scale.text(-0.13, 1.04, chr(ord("D") + column), transform=ax_scale.transAxes, fontsize=18, fontweight="bold", ha="left")


    method_handles = [Line2D([0], [0], color=method["color"], marker="o", linestyle="none", markersize=4, label=method["label"]) for method in FIXED_METHODS]
    marker_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=DARK_GRAY, linestyle="none", markersize=4, label="Mean standard error"),
        Line2D([0], [0], marker="s", markerfacecolor=DARK_GRAY, markeredgecolor="white", linestyle="none", markersize=3.5, label="SD of estimation error"),
        Line2D([0], [0], color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.8, label="Nominal 90% coverage"),
    ]
    fig.legend(handles=method_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94), columnspacing=0.8, handletextpad=0.35)
    fig.legend(handles=marker_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.025), columnspacing=1.1, handletextpad=0.4)
    fig.suptitle("Fixed-design interval coverage and error scale", y=0.99, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.16, top=0.80, wspace=0.20, hspace=0.52)

    outputs = save_figure(fig, "Figure_1")
    legend = {
        "claim": "Empirical coverage and error scale across the formal fixed-design scenarios and seeds; no direction is asserted separately from the plotted estimates.",
        "n": "Coverage dots use 1,000 planned test repetitions for each seed and scenario. Fit failures and method-specific invalid intervals are uncovered; exact fit-failure totals are in the source table.",
        "coverage_error": "Coverage bars show plus/minus one binomial Monte Carlo standard error from the formal summary.",
        "mean_se": "Open circles show the arithmetic mean of positive finite method-specific standard errors among coefficient-fit successes with valid variance. Exact per-point denominators are n_mean_standard_error in the source table and n_mean_standard_error in each exported plotted row.",
        "empirical_error_sd": "Filled squares show the sample standard deviation (ddof=1) of A_hat - A_true among coefficient-fit successes. Exact per-point denominators are n_empirical_error_sd in the source table and plotted row.",
        "seeds": "Each point in the lower panels is one of the three fixed-design seeds; colors identify methods.",
    }
    csv_path, json_path = write_source_bundle("fixed", frame, sources=[summary_path, *panel_paths], figure_legend=legend)
    update_manifest("figure_1_fixed_design", outputs, [summary_path, *panel_paths, csv_path, json_path])
    update_contract_status("figure_1_fixed_design", "rendered; visual QA pending")


def _real_coverage_columns(summary: pd.DataFrame) -> tuple[str, str, str, str | None]:
    row = summary.iloc[0]
    coverage_col = _first_present(row, ["coverage_planned_failures_uncovered", "coverage_planned", "coverage"])
    mcse_col = _first_present(row, ["coverage_mcse_planned", "coverage_mcse_1se", "coverage_mcse"])
    n_col = _first_present(row, ["n_planned", "n_test"])
    fail_col = _first_present(row, ["n_failed", "n_fit_fail", "fit_failures"], required=False)
    return coverage_col, mcse_col, n_col, fail_col


def build_real_figure() -> None:
    case_inputs = [
        {
            "case": "State design",
            "result_dir": RESULTS,
        },
        {
            "case": "City-matched design",
            "result_dir": DATA_ROOT / "city_case" / "results",
        },
    ]
    source_paths: list[Path] = []
    observed_frames: list[pd.DataFrame] = []
    formal_frames: list[pd.DataFrame] = []
    observed_specs: dict[str, pd.DataFrame] = {}
    observed_sensitivity_paths: dict[str, Path] = {}
    formal_coverage_rows: list[dict[str, Any]] = []

    for case_input in case_inputs:
        case_label = case_input["case"]
        result_dir = case_input["result_dir"]
        observed_path = result_dir / "real_sensitivity.csv"
        base_summary_path = result_dir / "real_formal_summary.csv"
        base_paired_path = result_dir / "real_formal_paired_differences.csv"
        quasi_summary_path = result_dir / "quasi_formal_summary.csv"
        quasi_paired_path = result_dir / "quasi_formal_paired_differences.csv"
        quasi_replications_path = result_dir / "quasi_formal_replications.csv"
        required = [observed_path, base_summary_path, base_paired_path, quasi_summary_path, quasi_paired_path]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Required observed/formal/quasi source is missing for {case_label}.")

        observed_sensitivity_paths[case_label] = observed_path
        observed = pd.read_csv(observed_path, encoding="utf-8-sig")
        observed["_method"] = observed["method"].map(lambda value: _method_for_name(value, OBSERVED_REAL_METHODS))
        observed = observed[observed["_method"].notna()].copy()
        observed["_method_key"] = observed["_method"].map(lambda method: method["key"])
        spec_cols = ["time_df_per_year", "max_lag_days", "common_trim_days", "lag_basis_dimension"]
        if not set(spec_cols).issubset(observed.columns):
            raise KeyError(f"Observed sensitivity data are missing specification columns: {spec_cols}")
        observed["_spec_key"] = observed[spec_cols].astype(str).agg("|".join, axis=1)
        specs = observed[spec_cols + ["_spec_key"]].drop_duplicates().copy()
        specs["time_df_per_year"] = pd.to_numeric(specs["time_df_per_year"], errors="raise")
        specs["max_lag_days"] = pd.to_numeric(specs["max_lag_days"], errors="raise")
        specs = specs.sort_values(["time_df_per_year", "max_lag_days", "lag_basis_dimension"]).reset_index(drop=True)
        if len(specs) != 9 or set(observed["_method_key"]) != {method["key"] for method in OBSERVED_REAL_METHODS} or len(observed) != 27:
            raise ValueError(f"Expected a 9-specification by 3-method observed grid for {case_label}.")
        if observed["n_outcomes"].astype(int).nunique() != 1 or int(observed["n_outcomes"].astype(int).iloc[0]) != 1805:
            raise ValueError(f"{case_label} observed grid must use the same 1,805 daily outcomes in every row.")
        if not np.allclose(pd.to_numeric(observed["interval_level"], errors="coerce"), 0.90, atol=1e-12):
            raise ValueError(f"{case_label} observed intervals must be nominal 90%.")
        specs["_spec_index"] = np.arange(len(specs))
        specs["_spec_label"] = specs.apply(
            lambda row: f"{int(float(row['time_df_per_year']))}/{int(float(row['max_lag_days']))}", axis=1
        )
        spec_indices = dict(zip(specs["_spec_key"], specs["_spec_index"]))
        spec_labels = dict(zip(specs["_spec_key"], specs["_spec_label"]))
        observed["_spec_index"] = observed["_spec_key"].map(spec_indices)
        observed["_spec_label"] = observed["_spec_key"].map(spec_labels)
        observed["case"] = case_label
        observed["record_type"] = "observed"
        observed_specs[case_label] = specs
        observed_frames.append(observed)

        base_summary = pd.read_csv(base_summary_path, encoding="utf-8-sig")
        quasi_summary = pd.read_csv(quasi_summary_path, encoding="utf-8-sig")
        base_summary["_case_label"] = case_label
        quasi_summary["_case_label"] = case_label
        formal = pd.concat([base_summary, quasi_summary], ignore_index=True, sort=False)
        formal["_method"] = formal["method"].map(lambda value: _method_for_name(value, REAL_METHODS))
        formal = formal[formal["_method"].notna()].copy()
        formal["_method_key"] = formal["_method"].map(lambda method: method["key"])
        scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(formal["scenario"].astype(str))]
        if len(scenarios) != 3:
            raise ValueError(f"Expected three formal scenarios for {case_label}, found {scenarios}.")
        for _, row in formal.iterrows():
            method = row["_method"]
            coverage = _first_numeric(row, ["coverage_planned_failures_uncovered", "coverage_planned", "coverage"])
            mcse = _first_numeric(row, ["coverage_mcse_planned", "coverage_mcse_1se", "coverage_mcse"])
            n_planned = int(float(row["n_planned"]))
            failures = _first_numeric(row, ["n_failed", "n_fit_fail", "fit_failures"], required=False)
            if n_planned != 1000:
                raise ValueError(f"Figure 2 expects n=1,000 formal repetitions for {case_label}.")
            formal_coverage_rows.append(
                {
                    "case": case_label,
                    "scenario": str(row["scenario"]),
                    "scenario_label": _scenario_label(str(row["scenario"])),
                    "method": method["key"],
                    "method_label": method["label"],
                    "coverage": coverage,
                    "coverage_mcse_1se": mcse,
                    "n_planned": n_planned,
                    "failures_counted_uncovered": int(failures) if failures is not None else np.nan,
                }
            )
        formal_frames.append(formal)
        source_paths.extend([observed_path, base_summary_path, base_paired_path, quasi_summary_path, quasi_paired_path])
        if quasi_replications_path.exists():
            source_paths.append(quasi_replications_path)

    observed = pd.concat(observed_frames, ignore_index=True)
    formal_frame = pd.DataFrame(formal_coverage_rows)
    expected_formal_cells = len(case_inputs) * 3 * len(REAL_METHODS)
    if len(formal_frame) != expected_formal_cells:
        raise ValueError(f"Expected {expected_formal_cells} formal coverage cells, found {len(formal_frame)}.")
    if formal_frame.duplicated(["case", "scenario", "method"]).any():
        raise ValueError("Duplicate formal coverage cells in Figure 2 input.")

    observed_source_rows: list[pd.DataFrame] = []
    for case_label, case_observed in zip([item["case"] for item in case_inputs], observed_frames):
        observed_source_rows.append(
            case_observed[
                [
                    "case",
                    "record_type",
                    "_spec_index",
                    "_spec_label",
                    "time_df_per_year",
                    "max_lag_days",
                    "common_trim_days",
                    "lag_basis_dimension",
                    "_method_key",
                    "method",
                    "n_outcomes",
                    "A_hat",
                    "standard_error",
                    "interval_level",
                    "interval_lo",
                    "interval_hi",
                    "converged",
                ]
            ].rename(
                columns={
                    "_spec_index": "specification_index",
                    "_spec_label": "specification_label",
                    "_method_key": "method_key",
                }
            )
        )
    observed_source = pd.concat(observed_source_rows, ignore_index=True)
    formal_frame["record_type"] = "formal_coverage"
    source_frame = pd.concat([observed_source, formal_frame], ignore_index=True, sort=False)
    source_frame.to_csv(QA / "figure_sources_real.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    scenario_order = ["model_compatible", "smooth_omitted_interaction", "correlated_overdispersed_ar1"]
    observed_offsets = {"model": -0.12, "hac14": 0.0, "hac28": 0.12}
    formal_offsets = {"model": -0.27, "hac14": -0.09, "hac28": 0.09, "quasi_poisson_pearson_scale": 0.27}
    case_order = [item["case"] for item in case_inputs]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.25), sharey="col")
    letters = ["A", "B", "C", "D"]
    for case_index, case in enumerate(case_order):
        case_observed = observed[observed["case"] == case]
        case_formal = formal_frame[formal_frame["case"] == case]
        specs = observed_specs[case].sort_values("_spec_index")
        ax_obs = axes[case_index, 0]
        ax_cov = axes[case_index, 1]
        for method in OBSERVED_REAL_METHODS:
            subset = case_observed[case_observed["_method_key"] == method["key"]].sort_values("_spec_index")
            x = subset["_spec_index"].to_numpy(float) + observed_offsets[method["key"]]
            center = pd.to_numeric(subset["A_hat"], errors="coerce").to_numpy(float)
            lo = pd.to_numeric(subset["interval_lo"], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(subset["interval_hi"], errors="coerce").to_numpy(float)
            converged = _bool_values(subset["converged"]).to_numpy(bool)
            valid = converged & np.isfinite(center) & np.isfinite(lo) & np.isfinite(hi)
            ax_obs.errorbar(
                x[valid],
                center[valid],
                yerr=[center[valid] - lo[valid], hi[valid] - center[valid]],
                fmt=method["marker"],
                markersize=2.8,
                markerfacecolor=method["color"],
                markeredgecolor="white",
                markeredgewidth=0.4,
                ecolor=method["color"],
                elinewidth=0.7,
                capsize=1.7,
                zorder=3,
            )
        ax_obs.axhline(0, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.7, zorder=0)
        ax_obs.set_xticks(
            np.arange(len(specs)),
            [f"{int(float(row.time_df_per_year))}/{int(float(row.max_lag_days))}" for row in specs.itertuples()],
        )
        ax_obs.set_xlim(-0.5, len(specs) - 0.5)
        ax_obs.set_ylabel("Expected count contrast\n(factual − 1°C cooler path)")
        ax_obs.set_xlabel("Time df per year / maximum lag (days)")
        ax_obs.set_title(f"{('State', 'Municipality')[case_index]}: observed specifications", loc="left", pad=4, fontweight="bold")
        ax_obs.text(-0.10, 1.04, letters[case_index * 2], transform=ax_obs.transAxes, fontsize=18, fontweight="bold")
        ax_obs.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax_obs.grid(axis="y", color=GRID, linewidth=0.5)
        ax_obs.tick_params(axis="both", length=2.1, width=0.6, color=GRAY, labelsize=8)

        scenarios = [scenario for scenario in scenario_order if scenario in set(case_formal["scenario"])]
        if len(scenarios) != 3:
            raise ValueError(f"Expected three formal scenarios for {case}; found {scenarios}.")
        for scenario_index, scenario in enumerate(scenarios):
            group = case_formal[case_formal["scenario"] == scenario]
            for method in REAL_METHODS:
                subset = group[group["method"] == method["key"]]
                if len(subset) != 1:
                    raise ValueError(f"Missing or duplicate formal cell: {case}/{scenario}/{method['key']}.")
                row = subset.iloc[0]
                ax_cov.errorbar(
                    scenario_index + formal_offsets[method["key"]],
                    row["coverage"],
                    yerr=row["coverage_mcse_1se"],
                    fmt=method["marker"],
                    markersize=3.2,
                    color=method["color"],
                    ecolor=method["color"],
                    elinewidth=0.7,
                    capsize=1.7,
                    zorder=3,
                )
        ax_cov.axhline(0.90, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.7, zorder=0)
        ax_cov.set_ylim(0.0, 1.01)
        ax_cov.set_xlim(-0.55, 2.55)
        compact_scenarios = {"model_compatible": "Compatible", "smooth_omitted_interaction": "Omitted\ninteraction", "correlated_overdispersed_ar1": "Correlated\nAR(1)"}
        ax_cov.set_xticks(np.arange(3), [compact_scenarios[value] for value in scenarios], rotation=0, ha="center")
        ax_cov.set_ylabel("Coverage (planned n)")
        ax_cov.set_xlabel("Simulation scenario")
        ax_cov.set_title(f"{('State', 'Municipality')[case_index]}: simulation coverage", loc="left", pad=4, fontweight="bold")
        ax_cov.text(-0.10, 1.04, letters[case_index * 2 + 1], transform=ax_cov.transAxes, fontsize=18, fontweight="bold")
        ax_cov.grid(axis="y", color=GRID, linewidth=0.5)
        ax_cov.tick_params(axis="both", length=2.1, width=0.6, color=GRAY, labelsize=8)

    method_handles = [
        Line2D([0], [0], color=method["color"], marker=method["marker"], markersize=3.8, linestyle="none", label=method["label"])
        for method in REAL_METHODS
    ]
    method_handles.append(Line2D([0], [0], color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.8, label="Nominal 90% / zero"))
    fig.legend(handles=method_handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.945), frameon=False, handletextpad=0.35, columnspacing=0.8)
    fig.suptitle("Observed specifications and simulation coverage", y=0.99, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.85, bottom=0.14, hspace=0.55, wspace=0.32)

    outputs = save_figure(fig, "Figure_3")
    legend = {
        "observed_n": "State and city observed-data grids each use the same 1,805 daily outcomes, 1°C temperature shift, 21-day common trim and nine predeclared specifications. The intervals are nominal 90%.",
        "observed_methods": "Observed grids contain Poisson, HAC14 and HAC28 standard-error methods. Point estimates are shared within specification; horizontal offsets separate interval marks only.",
        "formal_n": "State and city formal coverage points use 1,000 planned known-truth replications per scenario; failed fits count as uncovered.",
        "formal_methods": "Formal coverage compares unscaled Poisson, HAC14, HAC28 and Pearson-scaled Quasi-Poisson intervals.",
        "formal_uncertainty": "Coverage bars show plus/minus one Monte Carlo standard error for the binomial coverage estimate.",
        "interpretation": "Observed-data intervals and repeated-sampling coverage are shown separately; observed intervals do not establish causal identification.",
    }
    csv_path, json_path = write_source_bundle("real", source_frame, sources=source_paths, figure_legend=legend)
    update_manifest("figure_3_observed_and_formal", outputs, [*source_paths, csv_path, json_path])
    update_contract_status("figure_3_observed_and_formal", "rendered; visual QA pending")


def _first_numeric(row: pd.Series, names: list[str], *, required: bool = True) -> float | None:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    if required:
        raise KeyError(f"None of the expected nonmissing values were present: {names}")
    return None


def build_mechanism_figure() -> None:
    case_inputs = [
        {
            "case": "State aggregate",
            "summary": RESULTS / "mechanism_diagnostic.csv",
            "metadata": RESULTS / "mechanism_diagnostic.json",
            "replications": RESULTS / "mechanism_diagnostic_intervals.csv",
            "quasi_summary": RESULTS / "quasi_formal_summary.csv",
            "quasi_paired": RESULTS / "quasi_formal_paired_differences.csv",
        },
        {
            "case": "City-matched aggregate",
            "summary": DATA_ROOT / "city_case" / "results" / "mechanism_diagnostic.csv",
            "metadata": DATA_ROOT / "city_case" / "results" / "mechanism_diagnostic.json",
            "replications": DATA_ROOT / "city_case" / "results" / "mechanism_diagnostic_intervals.csv",
            "quasi_summary": DATA_ROOT / "city_case" / "results" / "quasi_formal_summary.csv",
            "quasi_paired": DATA_ROOT / "city_case" / "results" / "quasi_formal_paired_differences.csv",
        },
    ]
    frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    for entry in case_inputs:
        missing = [path for key, path in entry.items() if key != "case" and not path.exists()]
        if missing:
            raise FileNotFoundError(f"Mechanism diagnostic source is incomplete for {entry['case']}.")
        metadata = json.loads(entry["metadata"].read_text(encoding="utf-8"))
        role = str(metadata.get("analysis_role", "")).lower()
        if "ancillary known-truth" not in role:
            raise ValueError(f"Unexpected analysis role for {entry['case']}: {metadata.get('analysis_role')}")
        frame = pd.read_csv(entry["summary"], encoding="utf-8-sig")
        if len(frame) != 3:
            raise ValueError(f"Expected three known-truth scenarios for {entry['case']}, got {len(frame)}.")
        if not (pd.to_numeric(frame["n_planned"], errors="raise") == 1000).all():
            raise ValueError(f"Expected 1,000 planned formal replications per scenario for {entry['case']}.")
        if not (pd.to_numeric(frame["n_converged"], errors="raise") == 1000).all():
            raise ValueError(f"Expected 1,000 converged formal fits per scenario for {entry['case']}.")
        quasi = pd.read_csv(entry["quasi_summary"], encoding="utf-8-sig")
        if len(quasi) != 3 or not (quasi["method"].astype(str) == "quasi_poisson_pearson_scale").all():
            raise ValueError(f"Expected three Quasi-Poisson baseline rows for {entry['case']}.")
        if not (pd.to_numeric(quasi["n_planned"], errors="raise") == 1000).all() or not quasi["all_replay_checks_pass"].astype(bool).all():
            raise ValueError(f"Quasi-Poisson baseline replay checks failed for {entry['case']}.")
        quasi_ratio = dict(zip(quasi["scenario"], pd.to_numeric(quasi["mean_se_quasi_over_empirical_sd"], errors="raise")))
        frame["case"] = entry["case"]
        frame["quasi_poisson_se_over_empirical_sd"] = frame["scenario"].map(quasi_ratio)
        if frame["quasi_poisson_se_over_empirical_sd"].isna().any():
            raise ValueError(f"A Quasi-Poisson scale ratio is missing for {entry['case']}.")
        frame["oracle_scale_over_empirical_sd"] = (
            pd.to_numeric(frame["known_process_linearized_oracle_se"], errors="raise")
            / pd.to_numeric(frame["empirical_error_sd"], errors="raise")
        )
        frames.append(frame)
        source_paths.extend([entry["summary"], entry["metadata"], entry["replications"], entry["quasi_summary"], entry["quasi_paired"]])
    source_frame = pd.concat(frames, ignore_index=True)
    scenario_order = ["model_compatible", "smooth_omitted_interaction", "correlated_overdispersed_ar1"]
    if set(source_frame["scenario"].astype(str)) != set(scenario_order):
        raise ValueError("Mechanism diagnostic scenarios do not match the predeclared three-scenario set.")
    if source_frame.duplicated(["case", "scenario"]).any():
        raise ValueError("Duplicate case/scenario mechanism summary records.")
    source_frame["_scenario_order"] = source_frame["scenario"].map({value: i for i, value in enumerate(scenario_order)})
    source_frame = source_frame.sort_values(["case", "_scenario_order"]).drop(columns=["_scenario_order"]).reset_index(drop=True)
    source_frame.to_csv(QA / "figure_sources_mechanism.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    scenario_short = {
        "model_compatible": "Compatible",
        "smooth_omitted_interaction": "Omitted\ninteraction",
        "correlated_overdispersed_ar1": "Correlated\nAR(1)",
    }
    scenario_x = {name: index for index, name in enumerate(scenario_order)}
    case_order = ["State aggregate", "City-matched aggregate"]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.55))
    letters = ["A", "B", "C", "D", "E", "F"]
    for case_index, case in enumerate(case_order):
        rows = source_frame[source_frame["case"] == case]
        bias_ax, ratio_ax, coverage_ax = axes[case_index]
        for scenario in scenario_order:
            row = rows[rows["scenario"] == scenario].iloc[0]
            x = scenario_x[scenario]
            bias_ax.errorbar(
                x + 0.12,
                float(row["empirical_mean_error"]),
                yerr=float(row["mcse_mean_error"]),
                fmt="o",
                markersize=3.8,
                color=BLUE,
                ecolor=BLUE,
                elinewidth=0.85,
                capsize=2.2,
                zorder=3,
            )
            bias_ax.plot(
                x - 0.12,
                float(row["deterministic_projection_bias"]),
                marker="s",
                markersize=3.5,
                color=GRAY,
                linestyle="none",
                zorder=3,
            )

            ratios = [
                ("Poisson", float(row["mean_se_over_empirical_sd_model"]), GRAY, "o"),
                ("Quasi-Poisson", float(row["quasi_poisson_se_over_empirical_sd"]), LIGHT_GRAY, "P"),
                ("HAC14", float(row["mean_se_over_empirical_sd_hac14"]), BLUE, "s"),
                ("HAC28", float(row["mean_se_over_empirical_sd_hac28"]), ORANGE, "^"),
                ("Linearized oracle", float(row["oracle_scale_over_empirical_sd"]), DARK_GRAY, "D"),
            ]
            offsets = [-0.26, -0.13, 0.0, 0.13, 0.26]
            for (_, ratio, color, marker), offset in zip(ratios, offsets):
                ratio_ax.plot(
                    x + offset,
                    ratio,
                    marker=marker,
                    markersize=3.5,
                    color=color,
                    linestyle="none",
                    zorder=3,
                )

            coverage_points = [
                ("Oracle scale", "coverage_oracle_scale", "mcse_coverage_oracle_scale", GRAY, "o", -0.10),
                (
                    "Oracle scale + known bias",
                    "coverage_oracle_scale_and_known_bias",
                    "mcse_coverage_oracle_scale_and_known_bias",
                    BLUE,
                    "s",
                    0.10,
                ),
            ]
            for _, value_col, mcse_col, color, marker, offset in coverage_points:
                coverage_ax.errorbar(
                    x + offset,
                    float(row[value_col]),
                    yerr=float(row[mcse_col]),
                    fmt=marker,
                    markersize=3.6,
                    color=color,
                    ecolor=color,
                    elinewidth=0.8,
                    capsize=2.0,
                    zorder=3,
                )

        bias_ax.axhline(0, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.75, zorder=0)
        bias_ax.set_title(f"{('State', 'Municipality')[case_index]}: bias", loc="left", pad=4, fontweight="bold")
        bias_ax.set_ylabel("Bias / mean error\n(expected count contrast)")
        bias_ax.grid(axis="y", color=GRID, linewidth=0.5)
        bias_ax.tick_params(axis="both", length=2.2, width=0.6, color=GRAY)
        bias_ax.text(-0.13, 1.04, letters[case_index * 3], transform=bias_ax.transAxes, fontsize=18, fontweight="bold")

        ratio_ax.axhline(1.0, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.75, zorder=0)
        ratio_ax.set_title("SE / error SD", loc="left", pad=4, fontweight="bold")
        ratio_ax.set_ylabel("Mean SE / error SD")
        ratio_ax.grid(axis="y", color=GRID, linewidth=0.5)
        ratio_ax.tick_params(axis="both", length=2.2, width=0.6, color=GRAY)
        ratio_ax.text(-0.13, 1.04, letters[case_index * 3 + 1], transform=ratio_ax.transAxes, fontsize=18, fontweight="bold")

        coverage_ax.axhline(0.90, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.75, zorder=0)
        coverage_ax.set_ylim(0.50, 1.00)
        coverage_ax.set_title("Oracle coverage", loc="left", pad=4, fontweight="bold")
        coverage_ax.set_ylabel("Coverage")
        coverage_ax.grid(axis="y", color=GRID, linewidth=0.5)
        coverage_ax.tick_params(axis="both", length=2.2, width=0.6, color=GRAY)
        coverage_ax.text(-0.13, 1.04, letters[case_index * 3 + 2], transform=coverage_ax.transAxes, fontsize=18, fontweight="bold")

        labels = [scenario_short[value] for value in scenario_order]
        for ax in (bias_ax, ratio_ax, coverage_ax):
            ax.set_xticks([0, 1, 2], labels, rotation=0, ha="center")
            ax.tick_params(axis="x", labelbottom=True)
        if case_index == 0:
            ratio_ax.set_ylim(
                0.35,
                max(
                    1.3,
                    float(
                        source_frame[
                            [
                                "mean_se_over_empirical_sd_model",
                                "quasi_poisson_se_over_empirical_sd",
                                "mean_se_over_empirical_sd_hac14",
                                "mean_se_over_empirical_sd_hac28",
                                "oracle_scale_over_empirical_sd",
                            ]
                        ].max().max()
                    )
                    + 0.08,
                ),
            )
        else:
            # The same range keeps the city row comparable with the state row.
            ratio_ax.set_ylim(axes[0, 1].get_ylim())

    bias_handles = [
        Line2D([0], [0], marker="s", color=GRAY, linestyle="none", markersize=4, label="Projection bias"),
        Line2D([0], [0], marker="o", color=BLUE, linestyle="none", markersize=4, label="Mean error ±MCSE"),
    ]
    ratio_handles = [
        Line2D([0], [0], marker="o", color=GRAY, linestyle="none", markersize=4, label="Poisson SE/SD"),
        Line2D([0], [0], marker="P", color=LIGHT_GRAY, linestyle="none", markersize=4, label="Quasi-Poisson SE/SD"),
        Line2D([0], [0], marker="s", color=BLUE, linestyle="none", markersize=4, label="HAC14 SE/SD"),
        Line2D([0], [0], marker="^", color=ORANGE, linestyle="none", markersize=4, label="HAC28 SE/SD"),
        Line2D([0], [0], marker="D", color=DARK_GRAY, linestyle="none", markersize=3.7, label="Oracle SE/SD"),
    ]
    coverage_handles = [
        Line2D([0], [0], marker="o", color=GRAY, linestyle="none", markersize=4, label="Oracle scale"),
        Line2D([0], [0], marker="s", color=BLUE, linestyle="none", markersize=4, label="Oracle + bias centering"),
    ]
    fig.legend(handles=bias_handles, loc="upper center", bbox_to_anchor=(0.17, 0.18), ncol=1, frameon=False, handletextpad=0.35, labelspacing=0.5)
    fig.legend(handles=ratio_handles, loc="upper center", bbox_to_anchor=(0.50, 0.18), ncol=2, frameon=False, handletextpad=0.35, columnspacing=0.55, labelspacing=0.5)
    fig.legend(handles=coverage_handles, loc="upper center", bbox_to_anchor=(0.82, 0.18), ncol=1, frameon=False, handletextpad=0.35, labelspacing=0.5)
    fig.suptitle("Sources of interval undercoverage", y=0.99, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.91, bottom=0.25, wspace=0.38, hspace=0.62)

    outputs = save_figure(fig, "Figure_4")
    figure_legend = {
        "status": "Ancillary known-truth diagnostic initiated after the state formal summary; it does not introduce or validate a new applied interval method.",
        "bias": "Gray squares are deterministic Poisson-score projection bias. Blue circles are empirical mean A_hat - A_true across n=1,000 formal replications; bars are plus/minus one Monte Carlo standard error from the supplied summary.",
        "scale": "Markers show the mean Poisson, Quasi-Poisson, HAC14 or HAC28 SE divided by empirical error SD, using the same formal replications. The linearized known-process oracle SE ratio is computed directly as oracle SE / empirical error SD. The dashed reference is 1.",
        "coverage": "Coverage benchmarks use a known-process oracle scale, with and without known-bias centering; bars are plus/minus one binomial Monte Carlo standard error. The dashed reference is nominal 0.90.",
        "scenarios": "Compatible, a shared omitted temporal term z_c × 0.12, and correlated overdispersed AR(1); each case has 1,000 planned and converged fits per scenario. This is the frozen mechanism diagnostic, not a new analysis.",
        "interpretation": "These diagnostics separate deterministic target bias, finite-sample mean error and error-scale mismatch. Oracle scale and known-bias centering are explanatory controls rather than deployable estimators.",
    }
    csv_path, json_path = write_source_bundle("mechanism", source_frame, sources=source_paths, figure_legend=figure_legend)
    update_manifest("figure_4_mechanism_diagnostic", outputs, [*source_paths, csv_path, json_path])
    update_contract_status("figure_4_mechanism_diagnostic", "rendered; visual QA pending")


def build_heatday_figure() -> None:
    """Plot only the formal heat-day test replicates for the five selected methods."""
    config_path = DATA_ROOT / "inputs" / "fixed_heatday_config.json"
    summary_path = RESULTS / "heatday_formal_summary.csv"
    reproducibility_path = RESULTS / "heatday_formal_reproducibility.json"
    structural_qa_path = RESULTS / "heatday_formal_structural_qa.json"
    required = [config_path, summary_path, reproducibility_path, structural_qa_path]
    required.extend(
        RESULTS / f"heatday_formal_panel_{scenario}_seed{seed}.csv"
        for scenario in HEATDAY_SCENARIO_ORDER
        for seed in (11501, 24799, 39119)
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Figure 5 requires the complete formal heat-day summary and all nine test-panel sources; "
            f"missing {relative_name(missing[0])}."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("study_id") != "fixed_heatday_attribution_v1" or config.get("protocol_status") != "approved":
        raise ValueError("Figure 5 requires the approved fixed heat-day protocol.")
    configured_scenarios = [item["name"] for item in config["design"]["scenarios"]]
    if configured_scenarios != HEATDAY_SCENARIO_ORDER:
        raise ValueError(f"Heat-day scenario order differs from the contract: {configured_scenarios}")
    seeds = [int(value) for value in config["design"]["fixed_design_seeds"]]
    if seeds != [11501, 24799, 39119]:
        raise ValueError(f"Unexpected fixed heat-day seed set: {seeds}")
    stage_counts = {
        key: int(value)
        for key, value in config["sampling"]["formal_reps_per_panel"].items()
    }
    expected_test_reps = int(stage_counts.get("test", 0))
    if expected_test_reps != 1000:
        raise ValueError(f"Figure 5 expects 1,000 planned formal test replicates, got {expected_test_reps}.")

    reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    if (
        reproducibility.get("mode") != "formal"
        or int(reproducibility.get("mode_id", -1)) != 92103
        or reproducibility.get("protocol_status") != "approved"
    ):
        raise ValueError("Heat-day reproducibility metadata is not from the approved formal run.")
    if reproducibility.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Heat-day formal config hash does not match the local protocol file.")
    structural_qa = json.loads(structural_qa_path.read_text(encoding="utf-8"))
    if (
        structural_qa.get("mode") != "formal"
        or int(structural_qa.get("mode_id", -1)) != 92103
        or int(structural_qa.get("panel_count", -1)) != 9
        or int(structural_qa.get("replication_rows", -1)) != 9 * sum(stage_counts.values())
        or int(structural_qa.get("summary_rows", -1)) != 99
        or not bool(structural_qa.get("truth_constant_within_each_panel"))
        or structural_qa.get("formal_reproducibility_sha256") != sha256_file(reproducibility_path)
    ):
        raise ValueError("Heat-day formal structural-QA evidence does not match the expected run.")

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    selected = summary.copy()
    selected["_method"] = selected["method"].map(lambda value: _method_for_name(value, FIXED_METHODS))
    selected = selected[selected["_method"].notna()].copy()
    selected["_method_key"] = selected["_method"].map(lambda method: method["key"])
    selected = selected[
        selected["scenario"].astype(str).isin(HEATDAY_SCENARIO_ORDER)
        & selected["_method_key"].isin([method["key"] for method in FIXED_METHODS])
    ].copy()
    if selected.empty:
        raise ValueError("No selected covariance methods were found in the heat-day formal summary.")

    coverage_col = _first_present(selected.iloc[0], ["coverage", "coverage_planned", "coverage_planned_failures_uncovered"])
    mcse_col = _first_present(selected.iloc[0], ["coverage_mcse_1se", "coverage_mcse_planned", "coverage_mcse"])
    n_col = _first_present(selected.iloc[0], ["n_planned", "n_test"])
    fail_col = _first_present(selected.iloc[0], ["n_fit_fail", "n_failed", "fit_failures"])
    invalid_col = _first_present(
        selected.iloc[0],
        ["n_method_interval_invalid_after_fit", "n_interval_invalid_after_fit"],
    )
    if set(selected["scenario"].astype(str)) != set(HEATDAY_SCENARIO_ORDER):
        raise ValueError("Heat-day formal summary does not contain all three predeclared scenarios.")
    if set(pd.to_numeric(selected["design_seed"], errors="raise").astype(int)) != set(seeds):
        raise ValueError("Heat-day formal summary does not contain the three predeclared design seeds.")
    if len(selected) != 3 * len(seeds) * len(FIXED_METHODS):
        raise ValueError(f"Expected 45 selected heat-day summary points, got {len(selected)}.")
    selected["_design_seed"] = pd.to_numeric(selected["design_seed"], errors="raise").astype(int)
    if selected.duplicated(["scenario", "_design_seed", "_method_key"]).any():
        raise ValueError("Duplicate heat-day formal records for scenario, design seed and method.")
    if not (pd.to_numeric(selected[n_col], errors="raise") == expected_test_reps).all():
        raise ValueError("A heat-day formal summary record has the wrong planned test denominator.")

    panel_paths = [
        RESULTS / f"heatday_formal_panel_{scenario}_seed{seed}.csv"
        for scenario in HEATDAY_SCENARIO_ORDER
        for seed in seeds
    ]
    panel_lookup: dict[tuple[str, int], pd.DataFrame] = {}
    for path in panel_paths:
        panel = pd.read_csv(path, encoding="utf-8-sig")
        if not {"mode", "mode_id", "scenario", "design_seed", "stage"}.issubset(panel.columns):
            raise ValueError(f"Missing run or stage metadata in {relative_name(path)}.")
        if not (panel["mode"].astype(str).str.lower() == "formal").all():
            raise ValueError(f"Non-formal panel rows found in {relative_name(path)}.")
        if not (pd.to_numeric(panel["mode_id"], errors="raise") == 92103).all():
            raise ValueError(f"Wrong formal mode ID in {relative_name(path)}.")
        scenario_values = panel["scenario"].astype(str).unique().tolist()
        seed_values = pd.to_numeric(panel["design_seed"], errors="raise").astype(int).unique().tolist()
        if len(scenario_values) != 1 or len(seed_values) != 1:
            raise ValueError(f"Panel identity varies within {relative_name(path)}.")
        scenario = scenario_values[0]
        seed = int(seed_values[0])
        observed_stage_counts = panel["stage"].astype(str).value_counts().to_dict()
        if observed_stage_counts != stage_counts:
            raise ValueError(
                f"Unexpected formal stage counts in {relative_name(path)}: {observed_stage_counts}"
            )
        test_rows = panel.loc[panel["stage"].astype(str) == "test"].copy()
        if scenario not in HEATDAY_SCENARIO_ORDER or seed not in seeds:
            raise ValueError(f"Unexpected scenario or seed in {relative_name(path)}.")
        if len(test_rows) != expected_test_reps:
            raise ValueError(f"Expected {expected_test_reps} test-stage rows in {relative_name(path)}.")
        if not {"coefficient_fit_ok", "error", "A_true"}.issubset(test_rows.columns):
            raise ValueError(f"Missing fit/error/truth columns in {relative_name(path)}.")
        if pd.to_numeric(test_rows["A_true"], errors="coerce").nunique(dropna=True) != 1:
            raise ValueError(f"A_true varies within {scenario}/seed {seed} test rows.")
        key = (scenario, seed)
        if key in panel_lookup:
            raise ValueError(f"Duplicate heat-day formal panel for {scenario}/seed {seed}.")
        panel_lookup[key] = test_rows
    if set(panel_lookup) != {(scenario, seed) for scenario in HEATDAY_SCENARIO_ORDER for seed in seeds}:
        raise ValueError("Heat-day formal panel sources do not cover every scenario and seed.")

    rows: list[dict[str, Any]] = []
    for _, summary_row in selected.iterrows():
        scenario = str(summary_row["scenario"])
        seed = int(summary_row["_design_seed"])
        method = summary_row["_method"]
        replications = panel_lookup[(scenario, seed)]
        fit_mask = _bool_values(replications["coefficient_fit_ok"])
        variance = pd.to_numeric(replications[method["variance"]], errors="coerce")
        valid_mask = fit_mask & np.isfinite(variance) & (variance > 0)
        if method["variance_ok"] in replications:
            valid_mask &= _bool_values(replications[method["variance_ok"]])
        se_values = np.sqrt(variance[valid_mask].to_numpy(float))
        error_values = pd.to_numeric(replications.loc[fit_mask, "error"], errors="coerce").to_numpy(float)
        error_values = error_values[np.isfinite(error_values)]
        if len(error_values) < 2:
            raise ValueError(f"Too few fitted test errors for {scenario}/seed {seed}.")

        n_planned = int(float(summary_row[n_col]))
        n_fit_failures = int(float(summary_row[fail_col]))
        n_invalid = int(float(summary_row[invalid_col]))
        calculated_fit_failures = int((~fit_mask).sum())
        calculated_invalid = int((fit_mask & ~valid_mask).sum())
        if n_fit_failures != calculated_fit_failures:
            raise ValueError(f"Fit-failure count mismatch for {scenario}/seed {seed}/{method['key']}.")
        if n_invalid != calculated_invalid:
            raise ValueError(f"Invalid-interval count mismatch for {scenario}/seed {seed}/{method['key']}.")

        raw_method = str(summary_row["method"])
        cover_col = f"cover__{raw_method}"
        if cover_col not in replications:
            raise KeyError(f"Missing replicate coverage column {cover_col}.")
        replicate_coverage = pd.to_numeric(replications[cover_col], errors="raise").to_numpy(float)
        coverage = float(summary_row[coverage_col])
        if not np.all(np.isfinite(replicate_coverage)) or not np.isclose(
            float(np.mean(replicate_coverage)), coverage, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Summary and test-row coverage disagree for {scenario}/seed {seed}/{method['key']}.")
        expected_mcse = math.sqrt(coverage * (1 - coverage) / n_planned)
        coverage_mcse = float(summary_row[mcse_col])
        if not np.isclose(expected_mcse, coverage_mcse, rtol=0, atol=1e-12):
            raise ValueError(f"Coverage MCSE does not use the planned denominator for {scenario}/seed {seed}.")

        rows.append(
            {
                "scenario": scenario,
                "scenario_label": HEATDAY_SCENARIO_LABELS[scenario],
                "design_seed": seed,
                "method": method["key"],
                "source_method": raw_method,
                "method_label": method["label"],
                "A_true_scenario_specific": float(pd.to_numeric(replications["A_true"], errors="raise").iloc[0]),
                "coverage": coverage,
                "coverage_mcse_1se": coverage_mcse,
                "n_planned_test": n_planned,
                "fit_failures_counted_uncovered": n_fit_failures,
                "invalid_method_intervals_after_fit_counted_uncovered": n_invalid,
                "mean_standard_error": float(np.mean(se_values)) if len(se_values) else np.nan,
                "n_mean_standard_error": int(len(se_values)),
                "empirical_error_sd": float(np.std(error_values, ddof=1)),
                "n_empirical_error_sd": int(len(error_values)),
                "coefficient_fit_successes": int(fit_mask.sum()),
                "panel_test_rows": len(replications),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(QA / "figure_sources_heatday.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.65), sharey="row")
    method_positions = np.arange(len(FIXED_METHODS), dtype=float)
    seed_offsets = np.linspace(-0.16, 0.16, len(seeds))
    for column, scenario in enumerate(HEATDAY_SCENARIO_ORDER):
        scenario_data = frame[frame["scenario"] == scenario]
        ax_cov = axes[0, column]
        for m_index, method in enumerate(FIXED_METHODS):
            subset = scenario_data[scenario_data["method"] == method["key"]].sort_values("design_seed")
            for seed_index, (_, point) in enumerate(subset.iterrows()):
                ax_cov.errorbar(
                    method_positions[m_index] + seed_offsets[seed_index],
                    point["coverage"],
                    yerr=point["coverage_mcse_1se"],
                    fmt="o",
                    markersize=3.5,
                    color=method["color"],
                    ecolor=method["color"],
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.9,
                    zorder=3,
                )
        ax_cov.axhline(0.90, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
        ax_cov.set_ylim(0.0, 1.01)
        ax_cov.set_xlim(-0.5, len(FIXED_METHODS) - 0.5)
        ax_cov.set_xticks(method_positions, [method["short"] for method in FIXED_METHODS], rotation=25, ha="right")
        ax_cov.set_ylabel("Coverage" if column == 0 else "")
        ax_cov.set_title(HEATDAY_SCENARIO_LABELS[scenario], loc="left", pad=4, fontweight="bold")
        ax_cov.text(-0.13, 1.04, chr(ord("A") + column), transform=ax_cov.transAxes, fontsize=18, fontweight="bold")
        ax_cov.grid(axis="y", color=GRID, linewidth=0.5)
        ax_cov.tick_params(axis="both", length=2.3, width=0.6, color=GRAY)

        ax_scale = axes[1, column]
        for m_index, method in enumerate(FIXED_METHODS):
            subset = scenario_data[scenario_data["method"] == method["key"]].sort_values("design_seed")
            for seed_index, (_, point) in enumerate(subset.iterrows()):
                xpos = method_positions[m_index] + seed_offsets[seed_index]
                ax_scale.plot(
                    xpos - 0.025,
                    point["mean_standard_error"],
                    marker="o",
                    markersize=3.0,
                    markerfacecolor="white",
                    markeredgecolor=method["color"],
                    markeredgewidth=0.9,
                    linestyle="none",
                    zorder=3,
                )
                ax_scale.plot(
                    xpos + 0.025,
                    point["empirical_error_sd"],
                    marker="s",
                    markersize=2.8,
                    markerfacecolor=method["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    linestyle="none",
                    zorder=3,
                )
        ax_scale.set_xlim(-0.5, len(FIXED_METHODS) - 0.5)
        ax_scale.set_xticks(method_positions, [method["short"] for method in FIXED_METHODS], rotation=25, ha="right")
        ax_scale.set_ylabel("Error scale\n(expected counts)" if column == 0 else "")
        ax_scale.grid(axis="y", color=GRID, linewidth=0.5)
        ax_scale.tick_params(axis="both", length=2.3, width=0.6, color=GRAY)
        ax_scale.text(-0.13, 1.04, chr(ord("D") + column), transform=ax_scale.transAxes, fontsize=18, fontweight="bold", ha="left")


    method_handles = [
        Line2D([0], [0], color=method["color"], marker="o", linestyle="none", markersize=4, label=method["label"])
        for method in FIXED_METHODS
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=DARK_GRAY, linestyle="none", markersize=4, label="Mean standard error"),
        Line2D([0], [0], marker="s", markerfacecolor=DARK_GRAY, markeredgecolor="white", linestyle="none", markersize=3.5, label="SD of estimation error"),
        Line2D([0], [0], color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.8, label="Nominal 90% coverage"),
    ]
    fig.legend(handles=method_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94), columnspacing=0.8, handletextpad=0.35)
    fig.legend(handles=marker_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.025), columnspacing=1.1, handletextpad=0.4)
    fig.suptitle("Heat-day fixed-design interval coverage and error scale", y=0.99, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.16, top=0.80, wspace=0.20, hspace=0.52)
    outputs = save_figure(fig, "Figure_2")
    figure_legend = {
        "claim": "The component-specific two-way CR1 interval undercovers relative to the legacy and corrected-common CR1 intervals in the model-compatible and source-quadratic heat-day designs; coverage varies sharply by method in the common-bias design, whose scenario-specific A_true changes.",
        "n": "Coverage points use 1,000 planned test-stage repetitions for each of three fixed-design seeds and scenario. Coefficient-fit failures and method-specific invalid intervals after a successful fit count as uncovered; fit-failure totals appear in each scenario panel and exact method-specific invalid counts are in the source table.",
        "coverage_error": "Coverage bars show plus/minus one binomial Monte Carlo standard error computed with the planned test denominator.",
        "mean_se": "Open circles show the arithmetic mean of positive finite method-specific standard errors among coefficient-fit successes with valid variance. The exact per-point denominator is n_mean_standard_error in the source table and plotted row.",
        "empirical_error_sd": "Filled squares show the sample standard deviation (ddof=1) of A_hat - A_true among coefficient-fit successes. Its exact per-point denominator is n_empirical_error_sd in the source table and plotted row, regardless of method-specific interval validity.",
        "replication_selection": "Each formal panel CSV contains separate bias-training, scale-calibration and test stages. This figure uses only the 1,000 planned rows with stage=test in each panel; no test-stage rows are sampled or dropped.",
        "seeds": "Each point in the coverage and error-scale panels is one of the three fixed-design seeds; colors identify the same five selected covariance methods as Figure 1.",
        "scenario_truth": "A_true is the expected count contrast sum(mu - mu_cf) for each fixed design. The common-bias heat-day scenario changes eta0 and therefore changes its scenario-specific truth. Coverage and A_hat - A_true use each design's own fixed truth; cross-scenario points are not same-truth comparisons. Outcome labels refer to cholecystectomy hospital admission/discharge records or all-cause registered deaths, as applicable.",
    }
    csv_path, json_path = write_source_bundle("heatday", frame, sources=[*required, *panel_paths], figure_legend=figure_legend)
    update_manifest("figure_2_heatday_fixed_design", outputs, [*required, *panel_paths, csv_path, json_path])
    update_contract_status("figure_2_heatday_fixed_design", "rendered; visual QA pending")


SENSITIVITY_PANEL_LABELS = {
    "BRA": "Cholecystectomy admission authorisations",
    "MEX": "Cholecystectomy discharges",
    "SIM": "All-cause registered deaths",
    "MEXD": "All-cause registered deaths",
}
SENSITIVITY_PANEL_ORDER = ["BRA", "MEX", "SIM", "MEXD"]
SENSITIVITY_DELTAS = [0.01, 0.025, 0.05, 0.10]
SENSITIVITY_WEIGHT_ORDER = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
SENSITIVITY_PANEL_COLORS = {
    "BRA": BLUE,
    "MEX": ORANGE,
    "SIM": DARK_GRAY,
    "MEXD": GRAY,
}
SENSITIVITY_PANEL_MARKERS = {"BRA": "o", "MEX": "s", "SIM": "^", "MEXD": "D"}


def _read_sensitivity_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    paths = [
        ROOT / "tables" / "sensitivity_bounds.csv",
        ROOT / "tables" / "sensitivity_in_class_checks.csv",
        ROOT / "tables" / "sensitivity_out_of_class.csv",
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required sensitivity table is missing: {relative_name(path)}")
    bounds, in_class, out_of_class = [pd.read_csv(path) for path in paths]
    if len(bounds) != 80:
        raise ValueError(f"Expected 80 full-grid sensitivity rows; found {len(bounds)}.")
    if len(in_class) != 16:
        raise ValueError(f"Expected 16 fixed-allocation checks; found {len(in_class)}.")
    if len(out_of_class) != 24:
        raise ValueError(f"Expected 24 out-of-class checks; found {len(out_of_class)}.")
    return bounds, in_class, out_of_class, paths


def build_sensitivity_grid_figure() -> None:
    bounds, _, _, input_paths = _read_sensitivity_tables()
    expected_pairs = {
        (panel, delta, lag0, lag1)
        for panel in SENSITIVITY_PANEL_ORDER
        for delta in SENSITIVITY_DELTAS
        for lag0, lag1 in SENSITIVITY_WEIGHT_ORDER
    }
    observed_pairs = {
        (str(row.panel), round(float(row.delta), 8), round(float(row.weight_lag0), 8), round(float(row.weight_lag1), 8))
        for row in bounds.itertuples(index=False)
    }
    expected_pairs = {
        (panel, round(delta, 8), round(lag0, 8), round(lag1, 8))
        for panel, delta, lag0, lag1 in expected_pairs
    }
    if observed_pairs != expected_pairs:
        raise ValueError("The sensitivity table does not contain the complete 4-panel × 5-weight × 4-delta grid.")
    if not bounds["feasible"].astype(bool).all() or not bounds["bounded_class"].astype(bool).all():
        raise ValueError("Figure S1 requires feasible rows from the stated bounded class.")
    if (bounds["base_half_width"] <= 0).any():
        raise ValueError("Figure S1 requires strictly positive base interval half-widths.")

    source_frame = bounds.copy()
    source_frame["outcome_label"] = source_frame["panel"].map(SENSITIVITY_PANEL_LABELS)
    source_frame["lag_allocation_label"] = source_frame.apply(
        lambda row: f"{row['weight_lag0']:.2f} / {row['weight_lag1']:.2f}", axis=1
    )
    source_frame["sensitivity_radius_over_base_half_width"] = (
        source_frame["sensitivity_radius"] / source_frame["base_half_width"]
    )
    if not np.isfinite(source_frame["sensitivity_radius_over_base_half_width"]).all():
        raise ValueError("Figure S1 contains a non-finite normalized sensitivity radius.")

    weight_labels = [f"{lag0:.2f} / {lag1:.2f}" for lag0, lag1 in SENSITIVITY_WEIGHT_ORDER]
    weight_labels_ordered = weight_labels
    color_max = max(1.6, math.ceil(float(source_frame["sensitivity_radius_over_base_half_width"].max()) * 10) / 10)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.8), sharex=True)
    image = None
    for ax, panel, letter in zip(axes.ravel(), SENSITIVITY_PANEL_ORDER, ["A", "B", "C", "D"]):
        subset = source_frame[source_frame["panel"] == panel]
        matrix = np.empty((len(weight_labels_ordered), len(SENSITIVITY_DELTAS)), dtype=float)
        for row_index, pair_label in enumerate(weight_labels_ordered):
            for column_index, delta in enumerate(SENSITIVITY_DELTAS):
                selected = subset[
                    (subset["lag_allocation_label"] == pair_label)
                    & np.isclose(subset["delta"], delta, atol=1e-12, rtol=0)
                ]
                if len(selected) != 1:
                    raise ValueError(f"Expected one Figure S1 cell for {panel}, weights={pair_label}, delta={delta}.")
                matrix[row_index, column_index] = float(selected["sensitivity_radius_over_base_half_width"].iloc[0])
        image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=color_max, interpolation="nearest")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if value >= color_max * 0.58 else DARK_GRAY,
                )
        panel_title = f"{panel}  |  {SENSITIVITY_PANEL_LABELS[panel]}"
        if panel == "BRA":
            panel_title = "BRA  |  Cholecystectomy admission\nauthorisations"
        ax.set_title(panel_title, loc="left", pad=4, fontweight="bold")
        ax.text(-0.11, 1.04, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")
        ax.set_xticks(np.arange(len(SENSITIVITY_DELTAS)), ["0.01", "0.025", "0.05", "0.10"])
        ax.set_yticks(np.arange(len(weight_labels_ordered)), weight_labels_ordered)
        if panel in {"MEX", "MEXD"}:
            ax.set_yticklabels([])
        ax.set_xlabel("δ (log-risk perturbation bound)")
        ax.grid(False)
        ax.tick_params(axis="both", length=0, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(1.2)
    axes[0, 0].set_ylabel("Lag 0 / lag 1 weights")
    axes[1, 0].set_ylabel("Lag 0 / lag 1 weights")
    assert image is not None
    colorbar_axis = fig.add_axes([0.30, 0.065, 0.40, 0.018])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Sensitivity radius / base 90% interval half-width", fontsize=8, labelpad=2)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=8, length=1.8)
    fig.suptitle("Monthly sensitivity across lag weights and δ", y=0.985, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.20, top=0.86, hspace=0.48, wspace=0.23)

    outputs = save_figure(fig, "Figure_S1")
    legend = {
        "claim": "The complete 4-panel × 5-allocation × 4-delta finite-grid display shows the deterministic sensitivity radius relative to the stored base 90% interval half-width.",
        "mark": "Each tile is sensitivity_radius divided by base_half_width; the printed value and blue intensity encode the same dimensionless ratio.",
        "grid": "All 80 panel × lag-weight × delta combinations are retained. Lag allocation labels give (weight for lag 0 / weight for lag 1), and the x-axis uses the predeclared delta values 0.01, 0.025, 0.05 and 0.10.",
        "class": "Rows represent the stated finite observed basis-grid and fixed-allocation perturbation class. This display is not an empirical or sampling-coverage result and does not assert a continuous-temperature certificate.",
        "target": "Expected count contrast is factual path minus the 1°C cooler counterfactual path. Cholecystectomy outcomes refer to admission authorisations or discharges; death outcomes are all-cause registered deaths.",
    }
    csv_path, json_path = write_source_bundle(
        "sensitivity_grid",
        source_frame,
        sources=[*input_paths, source_root() / 'additional_analyses/results/sensitivity_extension.json'],
        figure_legend=legend,
    )
    update_manifest("figure_S1_lagweight_delta", outputs, [*input_paths, csv_path, json_path])
    update_contract_status("figure_S1_lagweight_delta", "rendered; visual QA pending")


def build_deterministic_displacement_figure() -> None:
    _, in_class, out_of_class, input_paths = _read_sensitivity_tables()
    if not in_class["status"].eq("pass").all() or not in_class["all_candidates_feasible"].astype(bool).all():
        raise ValueError("Figure S2 requires passing within-class candidate feasibility checks.")
    if not in_class["all_candidates_within_class_bound"].astype(bool).all():
        raise ValueError("At least one within-class candidate exceeded its stored deterministic class bound.")
    if not (in_class["allocation_weights"].astype(str) == "[0.5, 0.5]").all():
        raise ValueError("Figure S2 within-class checks must use the fixed 0.5 / 0.5 allocation.")
    if not out_of_class["gap_within_remainder_bound"].astype(bool).all():
        raise ValueError("An out-of-class deterministic rowwise remainder check failed.")

    in_source = in_class.copy()
    in_source["outcome_label"] = in_source["panel"].map(SENSITIVITY_PANEL_LABELS)
    in_source["plotted_panel"] = "within_fixed_allocation_class"
    in_source["plotted_x"] = in_source["delta"].astype(float)
    in_source["plotted_y"] = (
        in_source["max_abs_exact_attribution_shift"] / in_source["bound_B_lin_plus_R_uniform"]
    )
    in_source["plotted_y_definition"] = "maximum exact target shift / stored uniform fixed-allocation bound"
    out_source = out_of_class.copy()
    out_source["outcome_label"] = out_source["panel"].map(SENSITIVITY_PANEL_LABELS)
    out_source["plotted_panel"] = "out_of_class_zero_sum_lag_transfer"
    out_source["plotted_x"] = out_source["t_maxrow_predictor_shift"].astype(float)
    out_source["plotted_y"] = out_source["absolute_exact_shift_over_delta_0_05_bound"].astype(float)
    out_source["plotted_y_definition"] = "absolute exact shift / delta 0.05 fixed-allocation envelope (scale comparison only)"
    source_frame = pd.concat([in_source, out_source], ignore_index=True, sort=False)
    within_ratio_max = float(in_source["plotted_y"].max())
    out_ratio_max = float(out_source["plotted_y"].max())
    if within_ratio_max >= 1.0:
        raise ValueError("A within-class exact target displacement is not below its stored class envelope.")
    if out_ratio_max > 0.05:
        raise ValueError("The out-of-class ratio exceeds the predeclared display scale ceiling of 0.05.")

    if not np.isfinite(out_source["t_maxrow_predictor_shift"].to_numpy(float)).all() or (out_source["t_maxrow_predictor_shift"] <= 0).any():
        raise ValueError("Figure S2 log-scaled t values must be finite and strictly positive.")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.7))
    ax = axes[0]
    for panel in SENSITIVITY_PANEL_ORDER:
        subset = in_source[in_source["panel"] == panel].sort_values("delta")
        ax.plot(
            subset["delta"].to_numpy(float),
            subset["plotted_y"].to_numpy(float),
            color=SENSITIVITY_PANEL_COLORS[panel],
            marker=SENSITIVITY_PANEL_MARKERS[panel],
            markersize=3.4,
            linewidth=0.9,
            label=panel,
            zorder=3,
        )
    ax.axhline(1.0, color=GRAY, linestyle=(0, (3, 2)), linewidth=0.75, zorder=0)
    if not np.isfinite(in_source["plotted_x"].to_numpy(float)).all() or (in_source["plotted_x"] <= 0).any():
        raise ValueError("Figure S2 log-scaled delta values must be finite and positive.")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(0.008, 0.125)
    ax.set_ylim(0.80, 1.025)
    ax.set_xticks(SENSITIVITY_DELTAS, ["0.01", "0.025", "0.05", "0.10"])
    ax.set_xlabel("δ (log-risk bound; log scale)")
    ax.set_ylabel("Max exact shift / uniform class bound")
    ax.set_title("Within the fixed 0.5 / 0.5 class", loc="left", pad=4, fontweight="bold")
    ax.text(-0.10, 1.04, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")
    ax.grid(axis="y", color=GRID, linewidth=0.5)

    ax = axes[1]
    for panel in SENSITIVITY_PANEL_ORDER:
        subset = out_source[out_source["panel"] == panel]
        ax.scatter(
            subset["t_maxrow_predictor_shift"].to_numpy(float),
            subset["absolute_exact_shift_over_delta_0_05_bound"].to_numpy(float),
            color=SENSITIVITY_PANEL_COLORS[panel],
            marker=SENSITIVITY_PANEL_MARKERS[panel],
            s=21,
            linewidths=0.25,
            edgecolors="white",
            label=panel,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(0.009, 0.45)
    ax.set_ylim(0.0, 0.05)
    ax.set_xticks([0.01, 0.025, 0.05, 0.1, 0.2, 0.4], ["0.01", "0.025", "0.05", "0.10", "0.20", "0.40"])
    ax.set_xlabel("Maximum row log-predictor shift, t (log scale)")
    ax.set_ylabel("Absolute exact shift / δ = 0.05 envelope\n(scale comparison only)")
    ax.set_title("Out-of-class zero-sum lag transfer", loc="left", pad=4, fontweight="bold")
    ax.text(-0.10, 1.04, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")
    ax.grid(axis="y", color=GRID, linewidth=0.5)

    handles = [
        Line2D(
            [0], [0], color=SENSITIVITY_PANEL_COLORS[panel], marker=SENSITIVITY_PANEL_MARKERS[panel],
            linestyle="-", linewidth=0.9, markersize=3.4, label=panel
        )
        for panel in SENSITIVITY_PANEL_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.963), columnspacing=1.0, handletextpad=0.35)
    fig.suptitle("Target changes under response perturbations", y=0.99, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.17, top=0.82, wspace=0.34)

    outputs = save_figure(fig, "Figure_S2")
    legend = {
        "claim": "Every one of the 16 fixed 0.5/0.5 within-class checks has its maximum exact target displacement below the stored deterministic class envelope; 24 separate zero-sum lag-transfer checks are displayed only as deterministic out-of-class scale comparisons.",
        "panel_a": "Four outcome series over δ = 0.01, 0.025, 0.05 and 0.10; each point is max_abs_exact_attribution_shift divided by bound_B_lin_plus_R_uniform. Each panel × δ row checks 272 deterministic candidates (4 polygon vertices, 12 edge-convex combinations, 256 predeclared radial directions).",
        "panel_b": "Twenty-four out-of-class zero-sum lag-transfer rows are plotted as t_maxrow_predictor_shift against absolute_exact_shift_over_delta_0_05_bound. The fixed-allocation envelope is shown only for scale comparison and does not cover this perturbation class; the observed ratios do not establish a bound for that class.",
        "deterministic_remainder": "All out-of-class rows pass the realized deterministic rowwise remainder check recorded in the source table.",
        "coverage_claim": "Neither panel is a sampling-coverage or empirical interval-coverage calculation.",
        "target": "Target displacement is measured on the expected count contrast scale: factual path minus the 1°C cooler counterfactual path.",
    }
    csv_path, json_path = write_source_bundle(
        "sensitivity_displacement",
        source_frame,
        sources=[*input_paths, source_root() / 'additional_analyses/results/sensitivity_extension.json'],
        figure_legend=legend,
    )
    update_manifest("figure_S2_deterministic_displacement", outputs, [*input_paths, csv_path, json_path])
    update_contract_status("figure_S2_deterministic_displacement", "rendered; visual QA pending")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["1", "2", "3", "4", "5", "6", "S1", "S2", "all"], default="all")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Read-only root containing the frozen source results and inputs.",
    )
    args = parser.parse_args()
    global DATA_ROOT, RESULTS
    DATA_ROOT = args.data_root.resolve()
    RESULTS = DATA_ROOT / "results"
    set_publication_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    def build_daily_mvn_figure() -> None:
        runpy.run_path(str(Path(__file__).with_name("plot_mvn_figure.py")))

    selected = {
        "1": [build_fixed_figure],
        "2": [build_heatday_figure],
        "3": [build_real_figure],
        "4": [build_mechanism_figure],
        "5": [build_daily_mvn_figure],
        "6": [build_monthly_figure],
        "S1": [build_sensitivity_grid_figure],
        "S2": [build_deterministic_displacement_figure],
        "all": [
            build_fixed_figure,
            build_heatday_figure,
            build_real_figure,
            build_mechanism_figure,
            build_daily_mvn_figure,
            build_monthly_figure,
            build_sensitivity_grid_figure,
            build_deterministic_displacement_figure,
        ],
    }.get(
        args.figure,
        [],
    )
    for build in selected:
        build()


if __name__ == "__main__":
    main()

