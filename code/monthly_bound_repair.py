"""Observed monthly S1 fixed-allocation sensitivity bounds for four aggregate panels.

The new class centers the two-column temperature basis at each panel's factual
median, uses explicit lag-0/lag-1 mappings, and bounds all supplied factual and
counterfactual rows. It is a conditional sensitivity analysis, not a coverage
calibration or a reproduction of the historical external-constraint artifact.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import fixed_design
from bounded_sensitivity import bounded_sensitivity, unrestricted_cumulative_nullspace_diagnostic

ROOT = source_root()
INPUT_DIR = ROOT / "inputs" / "monthly"
MANIFEST_PATH = INPUT_DIR / "manifest.json"
OLD_RESULTS_PATH = ROOT / "inputs" / "legacy" / "simulation" / "results" / "external_constraint.json"
OUTPUT_PATH = run_root() / "results" / "monthly_bound_repair.json"
PANELS = ("BRA", "MEX", "SIM", "MEXD")
PANEL_OUTCOMES = {
    "BRA": "cholecystectomy admissions",
    "MEX": "cholecystectomy admissions",
    "SIM": "deaths",
    "MEXD": "deaths",
}
DELTA_GRID = (0.01, 0.025, 0.05, 0.10)
MAIN_DELTA = 0.05
ALLOCATION_WEIGHTS = (0.5, 0.5)
COUNTERFACTUAL_SHIFT_C = 1.0
ALPHA = 0.10
REQUIRED_FIELDS = ("state", "year", "month", "t", "tmean", "logpop", "y")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _load_panel(panel: str, records: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    entry = records[panel]
    path = INPUT_DIR / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = sha256(path)
    if observed_hash.lower() != str(entry["sha256"]).lower():
        raise ValueError(f"{panel}: panel CSV hash mismatch")
    frame = pd.read_csv(path, dtype={"state": str})
    if tuple(frame.columns) != REQUIRED_FIELDS:
        raise ValueError(f"{panel}: unexpected fields {list(frame.columns)}")
    frame["state"] = frame["state"].astype(str).str.strip().map(
        lambda label: label.zfill(2) if label.isdigit() else label
    )
    for col in ("year", "month", "t"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(int)
    for col in ("tmean", "logpop", "y"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    if frame.empty or frame["state"].isna().any():
        raise ValueError(f"{panel}: empty input or missing region label")
    if not frame["month"].between(1, 12).all():
        raise ValueError(f"{panel}: month is outside 1..12")
    frame["calendar_index"] = frame["year"] * 12 + frame["month"] - 1
    if frame.duplicated(["state", "calendar_index"]).any():
        raise ValueError(f"{panel}: duplicate region-month keys")
    expected_t = frame["calendar_index"] - int(frame["calendar_index"].min())
    if not np.array_equal(frame["t"].to_numpy(), expected_t.to_numpy()):
        raise ValueError(f"{panel}: t is inconsistent with calendar year/month")
    frame = frame.sort_values(["state", "calendar_index"], kind="mergesort").reset_index(drop=True)

    grouped = frame.groupby("state", sort=False)
    frame["lag1_tmean"] = grouped["tmean"].shift(1)
    frame["previous_calendar_index"] = grouped["calendar_index"].shift(1)
    frame["calendar_step"] = frame["calendar_index"] - frame["previous_calendar_index"]
    first_rows = frame["previous_calendar_index"].isna()
    adjacent_rows = frame["calendar_step"].eq(1)
    gap_rows = frame["calendar_step"].gt(1)
    finite_rows = np.isfinite(frame[["tmean", "lag1_tmean", "logpop", "y"]]).all(axis=1)
    positive_offsets = np.isfinite(frame["logpop"]) & np.isfinite(np.exp(np.clip(frame["logpop"], -700, 700)))
    keep = adjacent_rows & finite_rows & positive_offsets
    used = frame.loc[keep].copy().reset_index(drop=True)
    if used.empty:
        raise ValueError(f"{panel}: no rows remain after calendar-lag alignment")
    if used["state"].nunique() != frame["state"].nunique():
        raise ValueError(f"{panel}: at least one region has no usable adjacent-month pair")

    states = sorted(frame["state"].unique().tolist())
    calendar_count = int(frame["calendar_index"].max() - frame["calendar_index"].min() + 1)
    expected_keys = len(states) * calendar_count
    observed_keys = int(frame[["state", "calendar_index"]].drop_duplicates().shape[0])
    gaps_by_region = {}
    for state, group in frame.groupby("state", sort=True):
        periods = group["calendar_index"].to_numpy()
        gaps_by_region[str(state)] = int(np.maximum(np.diff(periods) - 1, 0).sum())
    metadata = {
        "file": str(path.relative_to(ROOT)),
        "sha256": observed_hash,
        "input_rows": int(len(frame)),
        "used_rows": int(len(used)),
        "regions": int(frame["state"].nunique()),
        "period_start": f"{int(frame['year'].min()):04d}-{int(frame.loc[frame['calendar_index'].idxmin(), 'month']):02d}",
        "period_end": f"{int(frame['year'].max()):04d}-{int(frame.loc[frame['calendar_index'].idxmax(), 'month']):02d}",
        "calendar_months": calendar_count,
        "complete_region_month_keys_expected": expected_keys,
        "missing_region_month_keys": int(expected_keys - observed_keys),
        "first_rows_without_prior_calendar_month": int(first_rows.sum()),
        "rows_after_calendar_gaps_dropped": int((gap_rows | (frame["calendar_step"].notna() & ~adjacent_rows & ~first_rows)).sum()),
        "rows_with_nonfinite_required_values_dropped": int((adjacent_rows & ~finite_rows).sum()),
        "per_region_missing_calendar_months": gaps_by_region,
        "lag_alignment": "lag 1 is the same region's exposure at calendar month t-1; first rows and rows without an adjacent prior calendar month are dropped",
    }
    return used, metadata


def _basis(x: np.ndarray, knots: np.ndarray, tref: float) -> np.ndarray:
    raw = fixed_design.source.ns_basis(np.asarray(x, float), knots)[:, 1:]
    reference = fixed_design.source.ns_basis(np.asarray([tref], float), knots)[0, 1:]
    return raw - reference


def _build_design(data: pd.DataFrame, all_factual_tmean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tref = float(np.median(all_factual_tmean))
    knots = np.quantile(all_factual_tmean, [0.15, 0.50, 0.85]).astype(float)
    x0 = data["tmean"].to_numpy(float)
    x1 = data["lag1_tmean"].to_numpy(float)
    xcf0 = x0 - COUNTERFACTUAL_SHIFT_C
    xcf1 = x1 - COUNTERFACTUAL_SHIFT_C
    region_codes = pd.Categorical(data["state"], categories=sorted(data["state"].unique())).codes
    month = data["month"].to_numpy(int) - 1
    trend = data["t"].to_numpy(float) / 12.0
    n = len(data)
    # Source S1 ordering: intercept, trend, lag 0 (2 cols), lag 1 (2 cols),
    # eleven calendar-month indicators, and R-1 region indicators.
    common = [np.ones(n), trend, _basis(x0, knots, tref), _basis(x1, knots, tref)]
    nuisance = [np.column_stack([(month == m).astype(float) for m in range(1, 12)])]
    region_count = int(data["state"].nunique())
    nuisance.append(np.column_stack([(region_codes == r).astype(float) for r in range(1, region_count)]))
    X = np.column_stack((*common, *nuisance))
    Xcf = X.copy()
    Xcf[:, 2:4] = _basis(xcf0, knots, tref)
    Xcf[:, 4:6] = _basis(xcf1, knots, tref)
    metadata = {
        "design_source": "c07_bai_frozen.design S1 two-lag ordering, with each temperature basis centered at the panel factual median",
        "basis_names": ["ns_linear_centered_at_Tref", "ns_nonlinear_centered_at_Tref"],
        "lag_blocks": [
            {"name": "lag0_current_month", "columns": [2, 3]},
            {"name": "lag1_previous_calendar_month", "columns": [4, 5]},
        ],
    }
    details = {
        "factual_median_reference_Tref_c": tref,
        "temperature_spline_knots_c": knots.tolist(),
        "knots_source": "15th, 50th, and 85th percentiles of all observed panel factual monthly temperatures before lag-row dropping",
        "design_columns": int(X.shape[1]),
        "lag_block_columns": [[2, 3], [4, 5]],
        "month_fixed_effect_columns": list(range(6, 17)),
        "region_fixed_effect_columns": list(range(17, X.shape[1])),
        "month_and_region_fixed_effects_identical_factual_counterfactual": bool(np.array_equal(X[:, 6:], Xcf[:, 6:])),
        "counterfactual_shift_c_applied_once_per_lag": COUNTERFACTUAL_SHIFT_C,
        "counterfactual_xlag0_min_c": float(np.min(xcf0)),
        "counterfactual_xlag0_max_c": float(np.max(xcf0)),
        "counterfactual_xlag1_min_c": float(np.min(xcf1)),
        "counterfactual_xlag1_max_c": float(np.max(xcf1)),
    }
    return X, Xcf, region_codes.astype(int), {**details, "metadata": metadata}


def _fit_panel(panel: str, records: dict[str, dict[str, Any]], old_results: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    full_data = pd.read_csv(INPUT_DIR / records[panel]["file"], dtype={"state": str})
    factual_tmean = pd.to_numeric(full_data["tmean"], errors="coerce").to_numpy(float)
    if not np.isfinite(factual_tmean).all():
        raise ValueError(f"{panel}: factual temperature has missing/nonfinite values")
    data, panel_qc = _load_panel(panel, records)
    X, Xcf, region, design_details = _build_design(data, factual_tmean)
    y = data["y"].to_numpy(float)
    offset = data["logpop"].to_numpy(float)
    year = data["year"].to_numpy(int)
    fit = fixed_design.fit_tw(y, X, offset, region, year, cfg)
    p = X.shape[1]
    full_rank = bool(fit["rank"] == p)
    if not fit["conv"] or not full_rank:
        raise RuntimeError(
            f"{panel}: fit did not meet convergence/rank checks "
            f"(converged={fit['conv']}, rank={fit['rank']}/{p}, max_step={fit['step']})"
        )
    beta = np.asarray(fit["beta"], float)
    eta = X @ beta + offset
    eta_cf = Xcf @ beta + offset
    if max(float(np.max(eta)), float(np.max(eta_cf))) >= math.log(np.finfo(float).max) - 2.0:
        raise OverflowError(f"{panel}: fitted predictor is too large for finite means")
    if min(float(np.min(eta)), float(np.min(eta_cf))) < -30 or max(float(np.max(eta)), float(np.max(eta_cf))) > 30:
        raise RuntimeError(f"{panel}: unclipped fitted predictors fall outside fixed_design.fit_tw's [-30, 30] range")
    mu = np.exp(eta)
    mucf = np.exp(eta_cf)
    A_hat = float(mu.sum() - mucf.sum())
    gradient = X.T @ mu - Xcf.T @ mucf
    variance = float(gradient @ fit["V_component"] @ gradient)
    if not np.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"{panel}: corrected component-specific CR1 attribution variance is not positive ({variance})")
    standard_error = math.sqrt(variance)
    df = int(fit["df"])
    tcrit = float(student_t.ppf(1.0 - ALPHA / 2.0, df))
    base_half_width = tcrit * standard_error
    metadata = design_details["metadata"]
    nullspace = unrestricted_cumulative_nullspace_diagnostic(beta, X, Xcf, offset, metadata)

    bounds = []
    for delta in DELTA_GRID:
        report = bounded_sensitivity(
            beta, X, Xcf, offset, metadata,
            weights=ALLOCATION_WEIGHTS,
            delta=delta,
            base_half_width=base_half_width,
            sample_count=64,
            seed=20261112,
        )
        bounds.append({"delta": delta, **report})
    selected = next(row for row in bounds if math.isclose(row["delta"], MAIN_DELTA, rel_tol=0, abs_tol=1e-15))

    old = old_results.get(panel)
    old_comparison = None
    if old is not None:
        old_comparison = {
            "outcome_label": PANEL_OUTCOMES[panel],
            "historical_artifact": {key: old.get(key) for key in ("A1", "A_lo", "A_hi", "robust_lo", "robust_hi", "robust_width", "base_width", "tipping_delta")},
            "historical_artifact_validity": "historical_artifact_unverified_statistical_validity",
            "comparison_status": "not a like-for-like width or interval comparison: historical S1 column mapping and sensitivity-class specification were flagged in statistical_code_review.md; the repaired class is newly centered and fixed-weight",
            "old_artifact_file": str(OLD_RESULTS_PATH.relative_to(ROOT)),
        }

    return {
        "panel": panel,
        "status": "ok",
        "outcome": {"input_field": "y", "label": PANEL_OUTCOMES[panel], "unit": "expected outcome count over retained months"},
        "panel_qc": panel_qc,
        "design": design_details,
        "fit": {
            "family_and_link": "Poisson log-link pseudo-likelihood with log population offset",
            "fit_method": "fixed_design.fit_tw; component-specific two-way region/year/intersection CR1 covariance",
            "fit_max_iterations": int(cfg["fit"]["max_iterations"]),
            "fit_tolerance_max_abs_step": float(cfg["fit"]["tolerance_max_abs_step"]),
            "converged": bool(fit["conv"]),
            "iterations": int(fit["iter"]),
            "max_abs_step": float(fit["step"]),
            "design_rank": int(fit["rank"]),
            "design_columns": int(p),
            "condition_number_information": float(fit["cond"]),
            "used_lstsq": bool(fit["lstsq"]),
            "used_pinv": bool(fit["pinv"]),
            "region_clusters": int(fit["nr"]),
            "year_clusters": int(fit["ny"]),
            "region_year_intersections": int(fit["nx"]),
            "df": df,
            "alpha": ALPHA,
            "critical_value_exact_student_t": tcrit,
            "corrected_attribution_variance": variance,
            "corrected_attribution_standard_error": standard_error,
            "base_half_width": base_half_width,
            "base_interval": [A_hat - base_half_width, A_hat + base_half_width],
            "base_interval_status": "model-based approximate interval; no exact coverage claim",
            "covariance_min_eigenvalue": float(fit["mineig_component"]),
        },
        "estimate": {
            "A_hat_expected_outcome_count": A_hat,
            "fitted_factual_expected_outcome_count": float(mu.sum()),
            "fitted_counterfactual_expected_outcome_count": float(mucf.sum()),
            "temperature_counterfactual_shift_c": COUNTERFACTUAL_SHIFT_C,
        },
        "sensitivity_class": {
            "class_id": "centered_two_lag_fixed_allocation_finite_grid_v1",
            "allocation_weights": list(ALLOCATION_WEIGHTS),
            "delta_grid_predeclared_in_script": list(DELTA_GRID),
            "main_delta": MAIN_DELTA,
            "constraints": "|b(T)'u| <= delta for all observed factual and counterfactual basis rows in both mapped lag blocks, centered at the fixed panel factual median Tref",
            "uniform_exponential_remainder": True,
            "continuous_temperature_certificate": False,
            "automatic_1_over_0_85_correction": False,
            "main_delta_result": selected,
            "delta_grid_results": bounds,
            "unrestricted_cumulative_nullspace_diagnostic": nullspace,
            "interpretation": "conditional sensitivity bound for this newly declared fixed-allocation class; no empirical coverage or causal-identification claim",
        },
        "historical_comparison": old_comparison,
    }


def run() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    records = {entry["panel"]: entry for entry in manifest}
    if set(records) != set(PANELS):
        raise ValueError(f"monthly manifest panel set differs: {sorted(records)}")
    for panel, entry in records.items():
        if entry.get("fields") != list(REQUIRED_FIELDS):
            raise ValueError(f"{panel}: manifest field order/schema mismatch")
    with OLD_RESULTS_PATH.open("r", encoding="utf-8-sig") as stream:
        old_results = json.load(stream)
    cfg = fixed_design.read_config()
    panel_results = {}
    for panel in PANELS:
        try:
            panel_results[panel] = _fit_panel(panel, records, old_results, cfg)
        except Exception as exc:
            panel_results[panel] = {
                "panel": panel,
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                "input_file": records[panel].get("file"),
                "input_sha256": records[panel].get("sha256"),
            }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(ROOT),
        "method": "Observed monthly S1 two-lag fit with a newly centered fixed-allocation sensitivity class",
        "new_class_id": "centered_two_lag_fixed_allocation_finite_grid_v1",
        "analysis_scope": "four existing aggregate monthly panels; no outcome resampling or empirical coverage calculation",
        "outcome_mapping": {panel: {"input_field": "y", "label": PANEL_OUTCOMES[panel], "unit": "expected outcome count over retained months"} for panel in PANELS},
        "predeclared_sensitivity_grid": list(DELTA_GRID),
        "main_delta": MAIN_DELTA,
        "allocation_weights": list(ALLOCATION_WEIGHTS),
        "counterfactual_shift_c": COUNTERFACTUAL_SHIFT_C,
        "automatic_1_over_0_85_correction": False,
        "monthly_input_manifest": {
            "file": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha256(MANIFEST_PATH),
            "panels": {panel: {"file": records[panel]["file"], "sha256": records[panel]["sha256"], "loader_sha256": records[panel]["loader_sha256"]} for panel in PANELS},
        },
        "historical_artifact_source": {
            "file": str(OLD_RESULTS_PATH.relative_to(ROOT)),
            "sha256": sha256(OLD_RESULTS_PATH),
            "validity": "historical_artifact_unverified_statistical_validity",
            "comparison_limitation": "different lag mapping and sensitivity class; old width is not interpreted as a comparator for the new bound",
        },
        "code": {
            "monthly_bound_repair_sha256": sha256(Path(__file__).resolve()),
            "fixed_design_sha256": sha256(Path(fixed_design.__file__).resolve()),
            "bounded_sensitivity_sha256": sha256(Path(__file__).resolve().with_name("bounded_sensitivity.py")),
            "pinned_core_sim_sha256": sha256(fixed_design.SOURCE_FILE),
            "fixed_config_sha256": sha256(fixed_design.CONFIG),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "panels": panel_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    failures = [panel for panel, result in panel_results.items() if result["status"] != "ok"]
    print(json.dumps({"output": str(OUTPUT_PATH), "panel_count": len(panel_results), "failed_panels": failures}, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"monthly bound repair failed for panels: {failures}")
    return report


if __name__ == "__main__":
    run()
