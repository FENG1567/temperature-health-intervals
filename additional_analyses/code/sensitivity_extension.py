"""Deterministic lag-allocation sensitivity analysis.

The old monthly analysis is imported read-only for its exact input loader,
design builder, and sensitivity calculation. The stored monthly JSON omits the
coefficient vectors, so this script restores beta and its covariance with the
same deterministic fit settings and checks those fit summaries against the
stored A-hat, df, and standard error before producing any extension outputs.
No outcome resampling or empirical coverage calculation is performed.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import csv
import hashlib
import importlib
import json
import logging
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any

# Keep import caches out of the archived read-only source tree.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import ConvexHull
from scipy.stats import t as student_t


NEW_ROOT = run_root() / 'additional_analyses'
OLD_ROOT = source_root()
OLD_CODE = OLD_ROOT / "code"
OLD_INPUT_DIR = OLD_ROOT / "inputs" / "monthly"
OLD_MANIFEST_PATH = OLD_INPUT_DIR / "manifest.json"
OLD_RESULTS_PATH = OLD_ROOT / "results" / "monthly_bound_repair.json"
OLD_REPAIR_SCRIPT = OLD_CODE / "monthly_bound_repair.py"
OLD_BOUNDED_SCRIPT = OLD_CODE / "bounded_sensitivity.py"
OLD_FIXED_DESIGN = OLD_CODE / "fixed_design.py"
CONFIG_SNAPSHOT_PATH = source_root() / 'additional_analyses' / "inputs" / "fixed_config_snapshot_20261112.json"
NEW_CODE_DIR = source_root() / "additional_analyses" / "code"
RESULTS_DIR = NEW_ROOT / "results"
TABLES_DIR = NEW_ROOT / "tables"
QA_DIR = NEW_ROOT / "qa"
LOG_DIR = NEW_ROOT / "logs"
JSON_OUTPUT = RESULTS_DIR / "sensitivity_extension.json"
BOUNDS_OUTPUT = TABLES_DIR / "sensitivity_bounds.csv"
BRIDGE_OUTPUT = TABLES_DIR / "sensitivity_bridge.csv"
OUT_OF_CLASS_OUTPUT = TABLES_DIR / "sensitivity_out_of_class.csv"
IN_CLASS_OUTPUT = TABLES_DIR / "sensitivity_in_class_checks.csv"
QA_OUTPUT = QA_DIR / "sensitivity_qa.json"
LOG_OUTPUT = LOG_DIR / "sensitivity_extension.log"
STDOUT_OUTPUT = LOG_DIR / "sensitivity_extension.stdout.log"
STDERR_OUTPUT = LOG_DIR / "sensitivity_extension.stderr.log"
LOCK_PATH = LOG_DIR / ".sensitivity_extension.lock"

PANELS = ("BRA", "MEX", "SIM", "MEXD")
WEIGHT_GRID = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0))
DELTA_GRID = (0.01, 0.025, 0.05, 0.10)
OUT_OF_CLASS_T_GRID = (0.01, 0.025, 0.05, 0.10, 0.20, 0.40)
OUT_OF_CLASS_BASE_DIRECTION = (1.0, 0.0)
PREDECLARED_DIRECTION_COUNT = 256
REGRESSION_REL_TOLERANCE = 1e-6
CHECK_TOLERANCE = 1e-9
SAMPLE_COUNT = 64
SAMPLE_SEED = 20261112
PRIOR_FAILED_PID = 72884
PRIOR_FAILURE_REASON = "The first hidden launch exited before creating the application log. A short preflight with the required bundled Python and no PYTHONPATH raised ModuleNotFoundError for scipy; importing the archived environment site-packages via PYTHONPATH then succeeded."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _regression_value(actual: Any, expected: Any, label: str, records: list[dict[str, Any]]) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    error = _relative_error(actual_value, expected_value)
    records.append({
        "label": label,
        "actual": actual_value,
        "expected": expected_value,
        "absolute_error": abs(actual_value - expected_value),
        "relative_error": error,
        "within_tolerance": bool(error <= REGRESSION_REL_TOLERANCE),
    })


def _compare_trees(expected: Any, actual: Any, label: str, summary: dict[str, Any]) -> None:
    """Compare a recomputed baseline report with its archived JSON object."""
    summary["compared_values"] += 1
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            summary["mismatch_count"] += 1
            if len(summary["mismatches"]) < 50:
                summary["mismatches"].append({"path": label, "expected_type": "object", "actual_type": type(actual).__name__})
            return
        expected_keys, actual_keys = set(expected), set(actual)
        if expected_keys != actual_keys:
            summary["mismatch_count"] += 1
            if len(summary["mismatches"]) < 50:
                summary["mismatches"].append({"path": label, "expected_keys_only": sorted(expected_keys - actual_keys), "actual_keys_only": sorted(actual_keys - expected_keys)})
        for key in sorted(expected_keys & actual_keys):
            _compare_trees(expected[key], actual[key], f"{label}.{key}", summary)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            summary["mismatch_count"] += 1
            if len(summary["mismatches"]) < 50:
                summary["mismatches"].append({"path": label, "expected_length": len(expected), "actual_length": len(actual) if isinstance(actual, list) else None})
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_trees(expected_item, actual_item, f"{label}[{index}]", summary)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        matched = isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
        if not matched:
            summary["mismatch_count"] += 1
            if len(summary["mismatches"]) < 50:
                summary["mismatches"].append({"path": label, "expected": expected, "actual": actual})
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        expected_value = float(expected)
        actual_value = float(actual)
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            # A non-finite scientific value never matches a finite archived value.
            matched = expected_value == actual_value
            if not matched:
                summary["mismatch_count"] += 1
                if len(summary["mismatches"]) < 50:
                    summary["mismatches"].append({"path": label, "expected": str(expected_value), "actual": str(actual_value), "reason": "non-finite mismatch"})
            return
        error = _relative_error(actual_value, expected_value)
        summary["max_relative_error"] = max(summary["max_relative_error"], error)
        # LP diagnostic residuals are near-zero floating-point values, not scientific count estimates.
        diagnostic = label.endswith(("LP_vertex_objective_max_error", "LP_vs_vertex_objective_error", "max_constraint_violation", "max_remainder_violation"))
        tolerated_roundoff = diagnostic and abs(actual_value - expected_value) <= 1e-10
        if error > REGRESSION_REL_TOLERANCE and not tolerated_roundoff:
            summary["mismatch_count"] += 1
            if len(summary["mismatches"]) < 50:
                summary["mismatches"].append({"path": label, "expected": expected_value, "actual": actual_value, "relative_error": error})
        return
    if actual != expected:
        summary["mismatch_count"] += 1
        if len(summary["mismatches"]) < 50:
            summary["mismatches"].append({"path": label, "expected": expected, "actual": actual})


def _setup_imports():
    if not OLD_ROOT.is_dir():
        raise FileNotFoundError(f"read-only source root is absent: {OLD_ROOT}")
    sys.path.insert(0, str(OLD_CODE))
    old_module = importlib.import_module("monthly_bound_repair")
    fixed_module = importlib.import_module("fixed_design")
    return old_module, fixed_module


def _json_diff_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else f".{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_json_diff_paths(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}.length"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_json_diff_paths(left_item, right_item, f"{path}[{index}]"))
        return differences
    return [] if left == right else [path or "."]


def _load_sources(old_module: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    with OLD_MANIFEST_PATH.open("r", encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    records = {entry["panel"]: entry for entry in manifest}
    if set(records) != set(PANELS):
        raise ValueError(f"monthly manifest panel set differs: {sorted(records)}")
    with OLD_RESULTS_PATH.open("r", encoding="utf-8-sig") as stream:
        original = json.load(stream)
    missing_panels = [panel for panel in PANELS if panel not in original.get("panels", {})]
    if missing_panels:
        raise KeyError(f"original monthly result JSON is missing panels: {missing_panels}")
    with CONFIG_SNAPSHOT_PATH.open("r", encoding="utf-8-sig") as stream:
        snapshot_config = json.load(stream)
    with (OLD_ROOT / "inputs" / "fixed_config.json").open("r", encoding="utf-8-sig") as stream:
        current_config = json.load(stream)
    snapshot_sha = _sha256_file(CONFIG_SNAPSHOT_PATH)
    archived_sha = str(original.get("code", {}).get("fixed_config_sha256", "")).lower()
    if not archived_sha or snapshot_sha.lower() != archived_sha:
        raise ValueError(f"new-root archived fixed-config snapshot hash mismatch: {snapshot_sha} != {archived_sha}")
    core_hash = _sha256_file(old_module.fixed_design.SOURCE_FILE)
    expected_core_hash = str(snapshot_config.get("source", {}).get("snapshot_sha256", "")).lower()
    if core_hash.lower() != expected_core_hash:
        raise ValueError(f"fixed-config snapshot pins core_sim hash {expected_core_hash}, observed {core_hash}")
    config_differences = _json_diff_paths(snapshot_config, current_config)
    fit_settings_equal = snapshot_config.get("fit") == current_config.get("fit")
    allowed_metadata_only_difference = bool(config_differences == [".protocol_status"] and fit_settings_equal)
    if not allowed_metadata_only_difference:
        raise ValueError(f"current fixed config differs from archived snapshot beyond its protocol_status metadata: differences={config_differences}; fit_settings_equal={fit_settings_equal}")
    config_comparison = {
        "archived_snapshot_file": str(CONFIG_SNAPSHOT_PATH),
        "archived_snapshot_sha256": snapshot_sha,
        "archived_expected_sha256": archived_sha,
        "current_approved_config_file": str(OLD_ROOT / "inputs" / "fixed_config.json"),
        "current_approved_config_sha256": _sha256_file(OLD_ROOT / "inputs" / "fixed_config.json"),
        "semantic_differences": config_differences,
        "fit_settings_equal": bool(fit_settings_equal),
        "difference_limited_to_protocol_metadata": allowed_metadata_only_difference,
    }
    return records, original, snapshot_config, config_comparison


def _validate_archived_hashes(records: dict[str, dict[str, Any]], original: dict[str, Any], config_comparison: dict[str, Any]) -> dict[str, Any]:
    code_expectations = {
        "monthly_bound_repair.py": (OLD_REPAIR_SCRIPT, "monthly_bound_repair_sha256"),
        "fixed_design.py": (OLD_FIXED_DESIGN, "fixed_design_sha256"),
        "bounded_sensitivity.py": (OLD_BOUNDED_SCRIPT, "bounded_sensitivity_sha256"),
        "pinned_core_sim.py": (OLD_ROOT / "inputs" / "code_snapshot" / "core_sim.py", "pinned_core_sim_sha256"),
    }
    checks: list[dict[str, Any]] = []
    archived_code = original.get("code", {})
    for name, (path, stored_key) in code_expectations.items():
        actual = _sha256_file(path)
        expected = archived_code.get(stored_key)
        checks.append({"name": name, "actual_sha256": actual, "archived_sha256": expected, "match": bool(expected and actual.lower() == str(expected).lower())})
    snapshot_expected = archived_code.get("fixed_config_sha256")
    checks.append({"name": "fixed_config_snapshot_20261112.json", "actual_sha256": config_comparison["archived_snapshot_sha256"], "archived_sha256": snapshot_expected, "match": bool(snapshot_expected and config_comparison["archived_snapshot_sha256"].lower() == str(snapshot_expected).lower())})
    manifest_actual = _sha256_file(OLD_MANIFEST_PATH)
    manifest_expected = original.get("monthly_input_manifest", {}).get("sha256")
    checks.append({"name": "monthly_input_manifest.json", "actual_sha256": manifest_actual, "archived_sha256": manifest_expected, "match": bool(manifest_expected and manifest_actual.lower() == str(manifest_expected).lower())})
    archived_panels = original.get("monthly_input_manifest", {}).get("panels", {})
    for panel in PANELS:
        path = OLD_INPUT_DIR / records[panel]["file"]
        actual = _sha256_file(path)
        expected = archived_panels.get(panel, {}).get("sha256")
        checks.append({"name": f"panel_{panel}.csv", "actual_sha256": actual, "archived_sha256": expected, "match": bool(expected and actual.lower() == str(expected).lower())})
    mismatches = [item for item in checks if not item["match"] and (item["name"].startswith("panel_") or item["name"] in ["pinned_core_sim.py", "fixed_config_snapshot_20261112.json", "monthly_input_manifest.json"])]
    if mismatches:
        raise RuntimeError(f"archived source hash regression failed: {mismatches}")
    return {"all_match": all(item["match"] for item in checks), "scientific_inputs_match": True, "portable_code_hashes_differ_from_archive": [item for item in checks if not item["match"]], "check_count": len(checks), "checks": checks, "fixed_config_comparison": config_comparison}


def _restore_panel(panel: str, records: dict[str, dict[str, Any]], original: dict[str, Any], old_module: Any, fixed_module: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    original_panel = original["panels"][panel]
    if original_panel.get("status") != "ok":
        raise RuntimeError(f"{panel}: original stored panel status is not ok")
    full_path = OLD_INPUT_DIR / records[panel]["file"]
    full_data = pd.read_csv(full_path, dtype={"state": str})
    factual_tmean = pd.to_numeric(full_data["tmean"], errors="coerce").to_numpy(float)
    if not np.isfinite(factual_tmean).all():
        raise ValueError(f"{panel}: nonfinite factual temperature in panel input")
    data, panel_qc = old_module._load_panel(panel, records)
    X, Xcf, region, design_details = old_module._build_design(data, factual_tmean)
    y = data["y"].to_numpy(float)
    offset = data["logpop"].to_numpy(float)
    year = data["year"].to_numpy(int)

    # Same deterministic model fit as monthly_bound_repair._fit_panel; there is
    # no outcome resampling, altered model, or new optimization setting here.
    fit = fixed_module.fit_tw(y, X, offset, region, year, cfg)
    p = int(X.shape[1])
    if not fit["conv"] or int(fit["rank"]) != p:
        raise RuntimeError(f"{panel}: deterministic coefficient recovery failed convergence/rank checks")
    beta = np.asarray(fit["beta"], dtype=float)
    V_component = np.asarray(fit["V_component"], dtype=float)
    if beta.shape != (p,) or V_component.shape != (p, p):
        raise ValueError(f"{panel}: restored coefficient/covariance dimensions are inconsistent")

    eta_f = X @ beta + offset
    eta_cf = Xcf @ beta + offset
    if not (np.isfinite(eta_f).all() and np.isfinite(eta_cf).all()):
        raise ValueError(f"{panel}: restored fitted predictors are nonfinite")
    if max(float(np.max(eta_f)), float(np.max(eta_cf))) >= math.log(np.finfo(float).max) - 2.0:
        raise OverflowError(f"{panel}: restored predictor exceeds finite exponential range")
    mu = np.exp(eta_f)
    mucf = np.exp(eta_cf)
    A_hat = float(mu.sum() - mucf.sum())
    gradient = X.T @ mu - Xcf.T @ mucf
    variance = float(gradient @ V_component @ gradient)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError(f"{panel}: restored attribution variance is not positive")
    standard_error = math.sqrt(variance)
    df = int(fit["df"])
    tcrit = float(student_t.ppf(1.0 - float(old_module.ALPHA) / 2.0, df))
    computed_half_width = float(tcrit * standard_error)
    stored_fit = original_panel["fit"]
    stored_estimate = original_panel["estimate"]
    stored_A_hat = float(stored_estimate["A_hat_expected_outcome_count"])
    stored_se = float(stored_fit["corrected_attribution_standard_error"])
    stored_df = int(stored_fit["df"])
    stored_half_width = float(stored_fit["base_half_width"])

    primary_regression: list[dict[str, Any]] = []
    _regression_value(A_hat, stored_A_hat, f"{panel}.A_hat", primary_regression)
    _regression_value(standard_error, stored_se, f"{panel}.SE", primary_regression)
    _regression_value(df, stored_df, f"{panel}.df", primary_regression)
    _regression_value(computed_half_width, stored_half_width, f"{panel}.base_half_width", primary_regression)
    failed = [entry for entry in primary_regression if not entry["within_tolerance"]]
    if failed:
        raise RuntimeError(f"{panel}: restored A_hat/SE/df/base-half-width regression exceeds 1e-6 relative tolerance: {failed}")

    metadata = design_details["metadata"]
    return {
        "panel": panel,
        "data": data,
        "X": np.asarray(X, dtype=float),
        "Xcf": np.asarray(Xcf, dtype=float),
        "region": np.asarray(region, dtype=int),
        "year": year,
        "offset": offset,
        "beta": beta,
        "V_component": V_component,
        "metadata": metadata,
        "design_details": design_details,
        "panel_qc": panel_qc,
        "fit": fit,
        "mu": mu,
        "mucf": mucf,
        "eta_f": eta_f,
        "eta_cf": eta_cf,
        "A_hat": A_hat,
        "gradient": gradient,
        "standard_error": standard_error,
        "variance": variance,
        "df": df,
        "base_half_width": stored_half_width,
        "regression": primary_regression,
        "input_path": full_path,
        "input_hash": _sha256_file(full_path),
        "analysis_reports": {},
    }


def _support_matrix(panel: dict[str, Any]) -> np.ndarray:
    X = panel["X"]
    Xcf = panel["Xcf"]
    blocks = panel["metadata"]["lag_blocks"]
    chunks = []
    for block in blocks:
        cols = np.asarray(block["columns"], dtype=int)
        chunks.extend((X[:, cols], Xcf[:, cols]))
    return np.vstack(chunks)


def _exact_A(panel: dict[str, Any], d_beta: np.ndarray) -> float:
    X = panel["X"]
    Xcf = panel["Xcf"]
    beta = panel["beta"]
    offset = panel["offset"]
    eta_f = X @ (beta + d_beta) + offset
    eta_cf = Xcf @ (beta + d_beta) + offset
    if not (np.isfinite(eta_f).all() and np.isfinite(eta_cf).all()):
        raise ValueError(f"{panel['panel']}: nonfinite exact perturbed predictor")
    if max(float(np.max(eta_f)), float(np.max(eta_cf))) >= math.log(np.finfo(float).max) - 2.0:
        raise OverflowError(f"{panel['panel']}: exact perturbed predictor exceeds finite exponential range")
    with np.errstate(over="raise", invalid="raise"):
        return float(np.exp(eta_f).sum() - np.exp(eta_cf).sum())


def _d_beta(panel: dict[str, Any], weights: tuple[float, float], u: np.ndarray) -> np.ndarray:
    dbeta = np.zeros_like(panel["beta"])
    for weight, block in zip(weights, panel["metadata"]["lag_blocks"]):
        cols = np.asarray(block["columns"], dtype=int)
        dbeta[cols] = float(weight) * np.asarray(u, dtype=float)
    return dbeta


def _class_candidates(panel: dict[str, Any], delta: float) -> tuple[list[dict[str, Any]], np.ndarray, int, float]:
    C = _support_matrix(panel)
    class_info = importlib.import_module("bounded_sensitivity")._class_status(C, float(delta), CHECK_TOLERANCE)
    if not class_info["bounded"]:
        raise RuntimeError(f"{panel['panel']} delta={delta}: fixed-weight class is unbounded")
    vertices = np.asarray(class_info["vertices"], dtype=float)
    hull = ConvexHull(vertices)
    ordered = vertices[hull.vertices]
    candidates: list[dict[str, Any]] = []
    for index, vertex in enumerate(ordered):
        candidates.append({"u": vertex.copy(), "candidate_type": "polygon_vertex", "source_index": index})
    for index in range(len(ordered)):
        start = ordered[index]
        end = ordered[(index + 1) % len(ordered)]
        for fraction in (0.25, 0.5, 0.75):
            candidates.append({"u": (1.0 - fraction) * start + fraction * end, "candidate_type": "edge_convex_combination", "source_index": index, "edge_fraction": fraction})

    directions = np.arange(PREDECLARED_DIRECTION_COUNT, dtype=float) * (2.0 * np.pi / PREDECLARED_DIRECTION_COUNT)
    for index, angle in enumerate(directions):
        direction = np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=float)
        denominator = float(np.max(np.abs(C @ direction), initial=0.0))
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError(f"{panel['panel']} delta={delta}: cannot normalize predeclared direction {index}")
        candidates.append({"u": direction * (float(delta) / denominator), "candidate_type": "predeclared_radial_direction", "source_index": index, "angle_radians": float(angle)})

    max_constraint_excess_raw = max(float(np.max(np.abs(C @ item["u"]) - delta, initial=-np.inf)) for item in candidates)
    max_constraint_excess = max(0.0, max_constraint_excess_raw)
    if max_constraint_excess > CHECK_TOLERANCE * (1.0 + abs(float(delta))):
        raise RuntimeError(f"{panel['panel']} delta={delta}: deterministic class candidate violates |C @ u| <= delta by {max_constraint_excess:g}")
    return candidates, C, len(ordered), max_constraint_excess


def _in_class_exact_check(panel: dict[str, Any], delta: float) -> dict[str, Any]:
    report = panel["analysis_reports"][(0.5, 0.5, delta)]
    if not report["bounded_class"] or not report["feasible"]:
        raise RuntimeError(f"{panel['panel']} delta={delta}: bounded sensitivity report is not a feasible bounded class")
    candidates, C, vertex_count, max_constraint_excess = _class_candidates(panel, delta)
    weights = (0.5, 0.5)
    A0 = float(report["A_hat"])
    B_lin = float(report["B_lin"])
    R_uniform = float(report["R_uniform"])
    bound = B_lin + R_uniform
    max_abs_actual_shift = 0.0
    max_abs_gap = 0.0
    min_candidate_margin = float("inf")
    max_envelope_excess = float("-inf")
    max_direct_mapping_error = 0.0
    for candidate in candidates:
        u = np.asarray(candidate["u"], dtype=float)
        dbeta = _d_beta(panel, weights, u)
        exact_shift = _exact_A(panel, dbeta) - A0
        linear_shift = float(np.asarray(report["gradient_u"], dtype=float) @ u)
        exact_gap = exact_shift - linear_shift
        max_abs_actual_shift = max(max_abs_actual_shift, abs(exact_shift))
        max_abs_gap = max(max_abs_gap, abs(exact_gap))
        min_candidate_margin = min(min_candidate_margin, bound - abs(exact_shift))
        max_envelope_excess = max(max_envelope_excess, abs(exact_shift) - bound)
        Df = 0.5 * panel["X"][:, panel["metadata"]["lag_blocks"][0]["columns"]] + 0.5 * panel["X"][:, panel["metadata"]["lag_blocks"][1]["columns"]]
        Dcf = 0.5 * panel["Xcf"][:, panel["metadata"]["lag_blocks"][0]["columns"]] + 0.5 * panel["Xcf"][:, panel["metadata"]["lag_blocks"][1]["columns"]]
        direct_f = panel["X"] @ dbeta
        direct_cf = panel["Xcf"] @ dbeta
        mapped_f = Df @ u
        mapped_cf = Dcf @ u
        max_direct_mapping_error = max(max_direct_mapping_error, float(np.max(np.abs(direct_f - mapped_f), initial=0.0)), float(np.max(np.abs(direct_cf - mapped_cf), initial=0.0)))

    margin = bound - max_abs_actual_shift
    within = bool(max_envelope_excess <= CHECK_TOLERANCE * (1.0 + abs(bound)) and max_abs_gap <= R_uniform + CHECK_TOLERANCE * (1.0 + abs(R_uniform)) and max_constraint_excess <= CHECK_TOLERANCE * (1.0 + abs(delta)))
    if not within:
        raise RuntimeError(f"{panel['panel']} delta={delta}: exact in-class candidate exceeds B_lin+R or the remainder bound")
    return {
        "panel": panel["panel"],
        "allocation_weights": list(weights),
        "delta": float(delta),
        "candidate_count": int(len(candidates)),
        "polygon_vertex_count": int(vertex_count),
        "polygon_edge_convex_combination_count": int(3 * vertex_count),
        "predeclared_direction_count": int(PREDECLARED_DIRECTION_COUNT),
        "candidate_types": {"polygon_vertex": int(vertex_count), "edge_convex_combination": int(3 * vertex_count), "predeclared_radial_direction": int(PREDECLARED_DIRECTION_COUNT)},
        "constraint": "For every candidate u, max over support rows of |C @ u| must be <= delta; C contains observed factual and counterfactual spline rows for both lag blocks.",
        "max_constraint_excess": float(max_constraint_excess),
        "all_candidates_feasible": bool(max_constraint_excess <= CHECK_TOLERANCE * (1.0 + abs(delta))),
        "B_lin": B_lin,
        "R_uniform": R_uniform,
        "bound_B_lin_plus_R_uniform": float(bound),
        "max_abs_exact_attribution_shift": float(max_abs_actual_shift),
        "max_abs_exact_gap_from_linear": float(max_abs_gap),
        "max_excess_over_bound": float(max_envelope_excess),
        "minimum_candidate_margin": float(min_candidate_margin),
        "margin_bound_minus_max_exact_shift": float(margin),
        "max_design_mapping_error": float(max_direct_mapping_error),
        "all_candidates_within_class_bound": within,
        "status": "pass",
        "interpretation": "deterministic known-coefficient exact A(beta+d) checks over vertices, edge convex combinations, and 256 predeclared radial directions; not an empirical sampling-coverage calculation",
    }


def _out_of_class_checks(panel: dict[str, Any], fixed_class_bounds: dict[float, float]) -> list[dict[str, Any]]:
    blocks = panel["metadata"]["lag_blocks"]
    cols0 = np.asarray(blocks[0]["columns"], dtype=int)
    cols1 = np.asarray(blocks[1]["columns"], dtype=int)
    base_direction = np.asarray(OUT_OF_CLASS_BASE_DIRECTION, dtype=float)
    shift_at_unit = max(
        float(np.max(np.abs(panel["X"][:, cols0] @ base_direction - panel["X"][:, cols1] @ base_direction), initial=0.0)),
        float(np.max(np.abs(panel["Xcf"][:, cols0] @ base_direction - panel["Xcf"][:, cols1] @ base_direction), initial=0.0)),
    )
    if not np.isfinite(shift_at_unit) or shift_at_unit <= 0:
        raise RuntimeError(f"{panel['panel']}: canonical zero-sum direction has no finite predictor shift")
    v = base_direction / shift_at_unit
    eta_direction_f = panel["X"][:, cols0] @ v - panel["X"][:, cols1] @ v
    eta_direction_cf = panel["Xcf"][:, cols0] @ v - panel["Xcf"][:, cols1] @ v
    normalized_max = max(float(np.max(np.abs(eta_direction_f), initial=0.0)), float(np.max(np.abs(eta_direction_cf), initial=0.0)))
    if not math.isclose(normalized_max, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(f"{panel['panel']}: max-row predictor-shift normalization failed ({normalized_max})")
    A0 = float(panel["A_hat"])
    g0 = panel["X"][:, cols0].T @ panel["mu"] - panel["Xcf"][:, cols0].T @ panel["mucf"]
    g1 = panel["X"][:, cols1].T @ panel["mu"] - panel["Xcf"][:, cols1].T @ panel["mucf"]
    output: list[dict[str, Any]] = []
    for t_value in OUT_OF_CLASS_T_GRID:
        t_value = float(t_value)
        d_beta = np.zeros_like(panel["beta"])
        d_beta[cols0] = t_value * v
        d_beta[cols1] = -t_value * v
        exact_shift = _exact_A(panel, d_beta) - A0
        linear_shift = float((g0 - g1) @ (t_value * v))
        exact_gap = exact_shift - linear_shift
        predictor_shift_f = t_value * eta_direction_f
        predictor_shift_cf = t_value * eta_direction_cf
        abs_f = np.abs(predictor_shift_f)
        abs_cf = np.abs(predictor_shift_cf)
        with np.errstate(over="raise", invalid="raise"):
            remainder_bound = float(panel["mu"] @ (np.expm1(abs_f) - abs_f) + panel["mucf"] @ (np.expm1(abs_cf) - abs_cf))
        margin = remainder_bound - abs(exact_gap)
        within = bool(abs(exact_gap) <= remainder_bound + CHECK_TOLERANCE * (1.0 + abs(remainder_bound)))
        if not within:
            raise RuntimeError(f"{panel['panel']} t={t_value}: exact zero-sum nonlinear gap exceeds its rowwise remainder bound")
        bound_005 = float(fixed_class_bounds[0.05])
        bound_010 = float(fixed_class_bounds[0.10])
        output.append({
            "panel": panel["panel"],
            "perturbation_class": "out_of_class_zero_sum_lag_transfer",
            "base_direction": list(OUT_OF_CLASS_BASE_DIRECTION),
            "normalized_v": v.tolist(),
            "maxrow_unit_direction_scale": float(shift_at_unit),
            "t_maxrow_predictor_shift": t_value,
            "observed_maxrow_predictor_shift": float(t_value * normalized_max),
            "d_beta_lag0": (t_value * v).tolist(),
            "d_beta_lag1": (-t_value * v).tolist(),
            "exact_attribution_shift": float(exact_shift),
            "linear_attribution_shift": float(linear_shift),
            "exact_gap_from_linear": float(exact_gap),
            "absolute_exact_gap": float(abs(exact_gap)),
            "rowwise_remainder_bound": remainder_bound,
            "margin_bound_minus_absolute_gap": float(margin),
            "gap_within_remainder_bound": within,
            "fixed_allocation_bound_delta_0_05": bound_005,
            "absolute_exact_shift_over_delta_0_05_bound": (float(abs(exact_shift) / bound_005) if bound_005 > 0 else None),
            "absolute_exact_shift_within_delta_0_05_bound": bool(abs(exact_shift) <= bound_005 + CHECK_TOLERANCE * (1.0 + abs(bound_005))),
            "fixed_allocation_bound_delta_0_10": bound_010,
            "absolute_exact_shift_over_delta_0_10_bound": (float(abs(exact_shift) / bound_010) if bound_010 > 0 else None),
            "absolute_exact_shift_within_delta_0_10_bound": bool(abs(exact_shift) <= bound_010 + CHECK_TOLERANCE * (1.0 + abs(bound_010))),
            "interpretation": "fixed canonical lag0 linear-basis direction; t is max-row log-predictor shift; ratios compare the exact target change with the fixed-allocation envelopes at delta 0.05 and 0.10 for context only; the fixed-allocation envelope does not cover this zero-sum transfer; the realized perturbation has its own deterministic rowwise remainder check",
        })
    return output


def _bounds_row(panel: str, weights: tuple[float, float], delta: float, report: dict[str, Any]) -> dict[str, Any]:
    sens = report.get("A_sensitivity_bounds") or [None, None]
    interval = report.get("interval_bounds") or [None, None]
    return {
        "panel": panel,
        "weight_lag0": float(weights[0]),
        "weight_lag1": float(weights[1]),
        "delta": float(delta),
        "A_hat": report.get("A_hat"),
        "B_lin": report.get("B_lin"),
        "R_uniform": report.get("R_uniform"),
        "sensitivity_radius": report.get("sensitivity_radius"),
        "A_sensitivity_lower": sens[0],
        "A_sensitivity_upper": sens[1],
        "base_half_width": report.get("base_half_width"),
        "interval_lower": interval[0],
        "interval_upper": interval[1],
        "feasible": report.get("feasible"),
        "bounded_class": report.get("bounded_class"),
        "constraint_rank": report.get("constraint_rank"),
        "constraint_rows": report.get("constraint_rows"),
        "polytope_vertex_count": report.get("polytope_vertex_count"),
        "LP_vertex_objective_max_error": report.get("LP_vertex_objective_max_error"),
        "exact_class_scope": report.get("class_scope"),
    }


def _bridge_row(panel: str, weights: tuple[float, float], delta: float, report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    bounds = report.get("A_sensitivity_bounds") or [None, None]
    base_bounds = baseline.get("A_sensitivity_bounds") or [None, None]
    interval = report.get("interval_bounds") or [None, None]
    base_interval = baseline.get("interval_bounds") or [None, None]
    return {
        "panel": panel,
        "delta": float(delta),
        "weight_lag0": float(weights[0]),
        "weight_lag1": float(weights[1]),
        "reference_weight_lag0": 0.5,
        "reference_weight_lag1": 0.5,
        "B_lin": report.get("B_lin"),
        "reference_B_lin": baseline.get("B_lin"),
        "change_B_lin_from_half_half": (None if report.get("B_lin") is None or baseline.get("B_lin") is None else float(report["B_lin"] - baseline["B_lin"])),
        "R_uniform": report.get("R_uniform"),
        "reference_R_uniform": baseline.get("R_uniform"),
        "change_R_uniform_from_half_half": (None if report.get("R_uniform") is None or baseline.get("R_uniform") is None else float(report["R_uniform"] - baseline["R_uniform"])),
        "sensitivity_radius": report.get("sensitivity_radius"),
        "reference_sensitivity_radius": baseline.get("sensitivity_radius"),
        "change_radius_from_half_half": (None if report.get("sensitivity_radius") is None or baseline.get("sensitivity_radius") is None else float(report["sensitivity_radius"] - baseline["sensitivity_radius"])),
        "A_sensitivity_lower": bounds[0],
        "A_sensitivity_upper": bounds[1],
        "reference_A_sensitivity_lower": base_bounds[0],
        "reference_A_sensitivity_upper": base_bounds[1],
        "change_lower_from_half_half": (None if bounds[0] is None or base_bounds[0] is None else float(bounds[0] - base_bounds[0])),
        "change_upper_from_half_half": (None if bounds[1] is None or base_bounds[1] is None else float(bounds[1] - base_bounds[1])),
        "interval_lower": interval[0],
        "interval_upper": interval[1],
        "reference_interval_lower": base_interval[0],
        "reference_interval_upper": base_interval[1],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(value) for key, value in row.items()})
    temporary.replace(path)


def run() -> dict[str, Any]:
    started = _utc_now()
    old_module, fixed_module = _setup_imports()
    records, original, cfg, config_comparison = _load_sources(old_module)
    baseline_hash_checks = _validate_archived_hashes(records, original, config_comparison)
    logger = logging.getLogger("sensitivity_extension")
    logger.info("old_source_root=%s", OLD_ROOT)
    logger.info("new_output_root=%s", NEW_ROOT)
    logger.info("all archived source hashes match: %s", baseline_hash_checks["all_match"])

    panels: dict[str, dict[str, Any]] = {}
    primary_fit_checks: list[dict[str, Any]] = []
    for panel_name in PANELS:
        logger.info("restoring deterministic fit for panel=%s", panel_name)
        panel = _restore_panel(panel_name, records, original, old_module, fixed_module, cfg)
        panels[panel_name] = panel
        primary_fit_checks.extend(panel["regression"])
        logger.info("panel=%s restored A_hat=%.12g SE=%.12g df=%d", panel_name, panel["A_hat"], panel["standard_error"], panel["df"])

    # First regenerate every archived 0.5/0.5 sensitivity result with the
    # archived point-check settings. Do not proceed to the M5 extension unless
    # all 16 complete result trees match within the declared relative tolerance.
    full_report_checks: list[dict[str, Any]] = []
    full_report_summary = {"compared_bounds": 0, "compared_values": 0, "mismatch_count": 0, "max_relative_error": 0.0, "mismatches": []}
    original_bounds = 0
    for panel_name in PANELS:
        panel = panels[panel_name]
        stored_grid = original["panels"][panel_name]["sensitivity_class"]["delta_grid_results"]
        if len(stored_grid) != len(DELTA_GRID):
            raise ValueError(f"{panel_name}: original JSON does not contain all four predeclared deltas")
        for stored in stored_grid:
            delta = float(stored["delta"])
            if delta not in DELTA_GRID:
                raise ValueError(f"{panel_name}: unexpected original delta {delta}")
            recalculated = old_module.bounded_sensitivity(
                panel["beta"], panel["X"], panel["Xcf"], panel["offset"], panel["metadata"],
                weights=(0.5, 0.5), delta=delta,
                base_half_width=panel["base_half_width"],
                sample_count=SAMPLE_COUNT, seed=SAMPLE_SEED,
            )
            cmp = {"compared_values": 0, "mismatch_count": 0, "max_relative_error": 0.0, "mismatches": []}
            _compare_trees(stored, recalculated, f"{panel_name}.delta={delta}", cmp)
            full_report_summary["compared_bounds"] += 1
            full_report_summary["compared_values"] += cmp["compared_values"]
            full_report_summary["mismatch_count"] += cmp["mismatch_count"]
            full_report_summary["max_relative_error"] = max(full_report_summary["max_relative_error"], cmp["max_relative_error"])
            for mismatch in cmp["mismatches"]:
                if len(full_report_summary["mismatches"]) < 50:
                    full_report_summary["mismatches"].append(mismatch)
            original_bounds += 1
            full_report_checks.append({"panel": panel_name, "delta": delta, **cmp})
            if cmp["mismatch_count"]:
                logger.error("archived 0.5/0.5 regression mismatch panel=%s delta=%s mismatch_count=%s", panel_name, delta, cmp["mismatch_count"])
    if original_bounds != 16 or full_report_summary["mismatch_count"]:
        raise RuntimeError(f"archived 0.5/0.5 regression failed before extension: {full_report_summary}")
    logger.info("archived A_hat/SE/df checks and all %d 0.5/0.5 bound report comparisons passed", original_bounds)

    bounds_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    exact_checks: list[dict[str, Any]] = []
    result_grid: dict[str, dict[str, dict[str, Any]]] = {panel_name: {} for panel_name in PANELS}
    for panel_name in PANELS:
        panel = panels[panel_name]
        for weight_pair in WEIGHT_GRID:
            for delta in DELTA_GRID:
                logger.info("computing bound panel=%s weights=%s delta=%s", panel_name, weight_pair, delta)
                report = old_module.bounded_sensitivity(
                    panel["beta"], panel["X"], panel["Xcf"], panel["offset"], panel["metadata"],
                    weights=weight_pair, delta=float(delta),
                    base_half_width=panel["base_half_width"],
                    sample_count=SAMPLE_COUNT, seed=SAMPLE_SEED,
                )
                if not report["feasible"] or not report["bounded_class"] or report["B_lin"] is None or report["R_uniform"] is None:
                    raise RuntimeError(f"{panel_name} weights={weight_pair} delta={delta}: bounded sensitivity calculation did not return finite bounds")
                key = f"w{weight_pair[0]:g}_{weight_pair[1]:g}_d{delta:g}"
                panel["analysis_reports"][(weight_pair[0], weight_pair[1], delta)] = report
                result_grid[panel_name][key] = report
                bounds_rows.append(_bounds_row(panel_name, weight_pair, delta, report))
        for delta in DELTA_GRID:
            exact_checks.append(_in_class_exact_check(panel, float(delta)))

        for delta in DELTA_GRID:
            reference = panel["analysis_reports"][(0.5, 0.5, delta)]
            for weight_pair in WEIGHT_GRID:
                report = panel["analysis_reports"][(weight_pair[0], weight_pair[1], delta)]
                bridge_rows.append(_bridge_row(panel_name, weight_pair, float(delta), report, reference))

    if len(bounds_rows) != 80:
        raise RuntimeError(f"expected 80 bound rows but computed {len(bounds_rows)}")
    if len(exact_checks) != 16 or not all(item["all_candidates_within_class_bound"] for item in exact_checks):
        raise RuntimeError("one or more deterministic in-class exact checks failed")

    out_of_class_rows: list[dict[str, Any]] = []
    for panel_name in PANELS:
        fixed_class_bounds = {
            0.05: float(panels[panel_name]["analysis_reports"][(0.5, 0.5, 0.05)]["sensitivity_radius"]),
            0.10: float(panels[panel_name]["analysis_reports"][(0.5, 0.5, 0.10)]["sensitivity_radius"]),
        }
        out_of_class_rows.extend(_out_of_class_checks(panels[panel_name], fixed_class_bounds))
    if len(out_of_class_rows) != 24 or not all(row["gap_within_remainder_bound"] for row in out_of_class_rows):
        raise RuntimeError("one or more deterministic out-of-class remainder checks failed")
    in_class_csv_rows: list[dict[str, Any]] = []
    for check in exact_checks:
        csv_row = {key: value for key, value in check.items() if key != "candidate_types"}
        csv_row.update({f"candidate_count_{key}": value for key, value in check["candidate_types"].items()})
        in_class_csv_rows.append(csv_row)

    panel_payload: dict[str, Any] = {}
    for panel_name in PANELS:
        panel = panels[panel_name]
        panel_payload[panel_name] = {
            "status": "ok",
            "input": {
                "file": str(panel["input_path"]),
                "sha256": panel["input_hash"],
                "loader_metadata": panel["panel_qc"],
            },
            "design": {
                "details": panel["design_details"],
                "matrices": {
                    "X": _matrix_record(panel["X"]),
                    "Xcf": _matrix_record(panel["Xcf"]),
                    "offset": _matrix_record(panel["offset"]),
                    "region": _matrix_record(panel["region"]),
                    "year": _matrix_record(panel["year"]),
                },
            },
            "coefficient_recovery": {
                "method": "same deterministic fixed_design.fit_tw settings as archived monthly_bound_repair.py; no outcome resampling",
                "fit_config": cfg["fit"],
                "beta": panel["beta"].tolist(),
                "V_component": panel["V_component"].tolist(),
                "beta_sha256": _matrix_record(panel["beta"])["sha256"],
                "V_component_sha256": _matrix_record(panel["V_component"])["sha256"],
                "fit_converged": bool(panel["fit"]["conv"]),
                "fit_iterations": int(panel["fit"]["iter"]),
                "fit_max_abs_step": float(panel["fit"]["step"]),
                "fit_rank": int(panel["fit"]["rank"]),
                "df_recomputed": int(panel["df"]),
                "standard_error_recomputed": float(panel["standard_error"]),
                "attribution_variance_recomputed": float(panel["variance"]),
                "stored_base_half_width_used": float(panel["base_half_width"]),
                "stored_summary_regression": panel["regression"],
            },
            "A_hat": float(panel["A_hat"]),
            "bounds": result_grid[panel_name],
            "in_class_exact_check_by_delta": [item for item in exact_checks if item["panel"] == panel_name],
            "out_of_class_exact_checks": [item for item in out_of_class_rows if item["panel"] == panel_name],
        }

    source_paths = {
        "monthly_bound_repair.py": OLD_REPAIR_SCRIPT,
        "bounded_sensitivity.py": OLD_BOUNDED_SCRIPT,
        "fixed_design.py": OLD_FIXED_DESIGN,
        "monthly_input_manifest.json": OLD_MANIFEST_PATH,
        "monthly_bound_repair.json": OLD_RESULTS_PATH,
        "current_approved_fixed_config.json": OLD_ROOT / "inputs" / "fixed_config.json",
        "fixed_config_snapshot_20261112.json": CONFIG_SNAPSHOT_PATH,
        "pinned_core_sim.py": OLD_ROOT / "inputs" / "code_snapshot" / "core_sim.py",
        "archived_environment_requirements.lock.txt": OLD_ROOT / "environment" / "requirements.lock.txt",
        "sensitivity_extension.py": NEW_CODE_DIR / "sensitivity_extension.py",
    }
    source_hashes = {name: {"file": str(path), "sha256": _sha256_file(path)} for name, path in source_paths.items()}
    for panel_name in PANELS:
        source_hashes[f"panel_{panel_name}.csv"] = {"file": str(panels[panel_name]["input_path"]), "sha256": panels[panel_name]["input_hash"]}

    qa = {
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "status": "pass",
        "old_source_root_read_only": str(OLD_ROOT),
        "new_output_root": str(NEW_ROOT),
        "source_hashes": source_hashes,
        "archived_source_hash_regression": baseline_hash_checks,
        "fixed_config_comparison": config_comparison,
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "pythondontwritebytecode": bool(sys.dont_write_bytecode),
            "blas_thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "parameters": {
            "panels": list(PANELS),
            "weight_grid": [list(value) for value in WEIGHT_GRID],
            "delta_grid": list(DELTA_GRID),
            "expected_bound_count": 80,
            "sample_count_for_sensitivity_diagnostics": SAMPLE_COUNT,
            "sensitivity_diagnostic_seed": SAMPLE_SEED,
            "regression_relative_tolerance": REGRESSION_REL_TOLERANCE,
            "in_class_predeclared_direction_count": PREDECLARED_DIRECTION_COUNT,
            "out_of_class_base_direction": list(OUT_OF_CLASS_BASE_DIRECTION),
            "out_of_class_t_grid_maxrow_predictor_shift": list(OUT_OF_CLASS_T_GRID),
            "fit_settings": cfg["fit"],
        },
        "regression": {
            "A_hat_SE_df_and_base_half_width_checks": primary_fit_checks,
            "all_primary_fit_checks_pass": all(row["within_tolerance"] for row in primary_fit_checks),
            "archived_weight_half_half": {
                **full_report_summary,
                "all_16_bounds_compared": original_bounds == 16,
                "all_16_bounds_within_relative_tolerance": full_report_summary["mismatch_count"] == 0 and original_bounds == 16,
                "per_bound_comparison": full_report_checks,
            },
        },
        "in_class_exact_checks": {
            "count": len(exact_checks),
            "candidate_count_total": int(sum(item["candidate_count"] for item in exact_checks)),
            "all_candidates_feasible": all(item["all_candidates_feasible"] for item in exact_checks),
            "maximum_constraint_excess": float(max(item["max_constraint_excess"] for item in exact_checks)),
            "all_pass": all(item["all_candidates_within_class_bound"] for item in exact_checks),
            "records": exact_checks,
        },
        "out_of_class_checks": {
            "count": len(out_of_class_rows),
            "all_exact_gap_remainder_checks_pass": all(row["gap_within_remainder_bound"] for row in out_of_class_rows),
            "records": out_of_class_rows,
        },
        "coverage_claim": {
            "empirical_sampling_coverage_calculated": False,
            "interpretation": "All reported M5 exact checks are deterministic known-coefficient sensitivity calculations, not empirical confidence-interval coverage estimates.",
        },
        "prior_start_attempt": {
            "pid": PRIOR_FAILED_PID,
            "status": "failed_before_application_log_creation",
            "diagnosis": PRIOR_FAILURE_REASON,
        },
        "preflight_side_effect": {
            "path": str(OLD_CODE / "__pycache__" / "monthly_bound_repair.cpython-312.pyc"),
            "description": "The short old-module import preflight generated this Python bytecode cache in the archived code tree. The cleanup attempt was rejected by the filesystem policy; source .py/CSV/JSON files remain protected, and sys.dont_write_bytecode is enabled for the formal run.",
        },
        "runtime_preflight": {
            "status": "pass",
            "required_bundled_python": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "imported_packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
            "imported_archived_modules": ["monthly_bound_repair", "fixed_design", "bounded_sensitivity"],
        },
        "outputs": {
            "json": str(JSON_OUTPUT),
            "bounds_csv": str(BOUNDS_OUTPUT),
            "bridge_csv": str(BRIDGE_OUTPUT),
            "out_of_class_csv": str(OUT_OF_CLASS_OUTPUT),
            "in_class_checks_csv": str(IN_CLASS_OUTPUT),
            "qa_json": str(QA_OUTPUT),
            "log": str(LOG_OUTPUT),
            "launcher_stdout": str(STDOUT_OUTPUT),
            "launcher_stderr": str(STDERR_OUTPUT),
        },
    }
    result = {
        "created_at_utc": _utc_now(),
        "project": str(NEW_ROOT),
        "method": "Deterministic fixed-weight sensitivity extension across five lag allocations and four finite-grid delta values, with known-coefficient exact checks.",
        "analysis_scope": "Four archived monthly panels; coefficients and covariance deterministically restored with the archived fit settings because the archived JSON did not serialize beta; no outcome resampling.",
        "class_claim": "conditional deterministic sensitivity bounds for stated finite-grid fixed-allocation classes only",
        "coverage_claim": qa["coverage_claim"],
        "source_hashes": source_hashes,
        "environment": qa["environment"],
        "fixed_config_comparison": config_comparison,
        "parameters": qa["parameters"],
        "archived_regression": qa["regression"],
        "panels": panel_payload,
    }

    # Commit the requested deliverables only after all regression and exact
    # checks have passed. These are new-root outputs; the source tree is never
    # written by this script.
    _write_json(JSON_OUTPUT, result)
    _write_csv(BOUNDS_OUTPUT, bounds_rows)
    _write_csv(BRIDGE_OUTPUT, bridge_rows)
    _write_csv(OUT_OF_CLASS_OUTPUT, out_of_class_rows)
    _write_csv(IN_CLASS_OUTPUT, in_class_csv_rows)
    _write_json(QA_OUTPUT, qa)
    logger.info("completed outputs: bounds=%d bridge=%d in_class_checks=%d out_of_class_checks=%d", len(bounds_rows), len(bridge_rows), len(exact_checks), len(out_of_class_rows))
    logger.info("created %s", JSON_OUTPUT)
    logger.info("created %s", BOUNDS_OUTPUT)
    logger.info("created %s", BRIDGE_OUTPUT)
    logger.info("created %s", OUT_OF_CLASS_OUTPUT)
    logger.info("created %s", IN_CLASS_OUTPUT)
    logger.info("created %s", QA_OUTPUT)
    return {"status": "pass", "bounds": len(bounds_rows), "in_class_checks": len(exact_checks), "out_of_class_checks": len(out_of_class_rows)}


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG_OUTPUT, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    logger = logging.getLogger("sensitivity_extension")
    try:
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        logger.error("refusing duplicate run because lock exists: %s", LOCK_PATH)
        return 3
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\nstarted_at_utc={_utc_now()}\n")
        logger.info("pid=%s", os.getpid())
        summary = run()
        logger.info("final_summary=%s", json.dumps(summary, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.error("run_failed=%s: %s", type(exc).__name__, exc)
        logger.error("traceback=%s", traceback.format_exc())
        return 1
    finally:
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            logger.exception("failed_to_remove_lock=%s", LOCK_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
