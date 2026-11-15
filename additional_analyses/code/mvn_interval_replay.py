#!/usr/bin/env python3
"""Replay the frozen daily state/city outcomes and compare MVN intervals.

Archived inputs are read-only. Rerun outputs use CA_RUN_ROOT.
Formal execution requires the explicit --formal-approved flag.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

OLD_ROOT = source_root()
NEW_ROOT = run_root() / 'additional_analyses'
NEW_ROOT.mkdir(parents=True,exist_ok=True)
EXPECTED_PYTHON = Path(sys.executable)

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
from scipy.stats import norm

PILOT_REPLICATIONS_PER_SCENARIO_MAX = 4
FORMAL_REPLICATIONS_PER_SCENARIO = 1000
PRIMARY_DRAWS = 2048
STABILITY_DRAWS = (4096, 8192)
STABILITY_REPLICATES = 20
PSD_RELATIVE_TOLERANCE = 1e-10
DRAW_SEED_TAG = 78111
CHECKPOINT_EVERY = 100
_WORKER_CONTEXT: dict[str, Any] | None = None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8")
    temp.replace(path)


def _legacy_module():
    source = OLD_ROOT / "code" / "quasi_baseline.py"
    spec = importlib.util.spec_from_file_location("mvn_legacy_quasi_baseline", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen replay module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_formal_rows(case_state: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    formal = case_state["formal"]
    config = case_state["config"]
    sim = config["simulation"]
    scenario_ids = case_state["scenario_ids"]
    expected_count = len(scenario_ids) * FORMAL_REPLICATIONS_PER_SCENARIO
    if len(formal) != expected_count:
        raise ValueError(f"Expected {expected_count} frozen formal rows, found {len(formal)}.")
    original_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in formal.to_dict(orient="records"):
        scenario = str(row["scenario"])
        replicate = int(row["replicate"])
        expected_seed = [
            int(sim["base_seed"]),
            int(sim["mode_ids"]["formal"]),
            int(scenario_ids[scenario]),
            replicate,
        ]
        stored_seed = json.loads(row["seed_entropy"])
        if stored_seed != expected_seed:
            raise ValueError(
                f"Frozen seed mismatch for {scenario} replicate {replicate}: "
                f"{stored_seed} != {expected_seed}"
            )
        key = (scenario, replicate)
        if key in original_rows:
            raise ValueError(f"Duplicate frozen row identifier: {key}")
        original_rows[key] = row
    if len(original_rows) != expected_count:
        raise ValueError("Frozen formal replication identifiers are not unique.")
    return original_rows


def _init_process(case: str) -> None:
    global _WORKER_CONTEXT
    legacy = _legacy_module()
    case_state = legacy._prepare_case(case)
    original_rows = _validate_formal_rows(case_state)
    legacy._WORKER = {"case_state": case_state, "original_rows": original_rows}
    _WORKER_CONTEXT = {"legacy": legacy, "case_state": case_state}


def _stable_factor(covariance: np.ndarray) -> tuple[np.ndarray | None, float, float, bool]:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        return None, float("nan"), float("nan"), False
    symmetric = (covariance + covariance.T) / 2.0
    if not np.all(np.isfinite(symmetric)):
        return None, float("nan"), float("nan"), False
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not len(eigenvalues) or not np.all(np.isfinite(eigenvalues)):
        return None, float("nan"), float("nan"), False
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = PSD_RELATIVE_TOLERANCE * scale
    minimum = float(np.min(eigenvalues))
    if minimum < -tolerance:
        return None, minimum, tolerance, False
    # Clip only negative eigenvalues within the declared numerical tolerance.
    root_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    factor = eigenvectors * root_values[None, :]
    return factor, minimum, tolerance, True


def _antithetic_indices(draws: int, maximum_draws: int) -> np.ndarray:
    half = draws // 2
    max_half = maximum_draws // 2
    return np.concatenate((np.arange(half), max_half + np.arange(half)))


def _exact_nonlinear_draws(
    design_factor: np.ndarray,
    cf_design_factor: np.ndarray,
    eta_factual: np.ndarray,
    eta_counterfactual: np.ndarray,
    z_draws: np.ndarray,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    count = z_draws.shape[1]
    values = np.full(count, np.nan, dtype=float)
    valid = np.zeros(count, dtype=bool)
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        z = z_draws[:, start:end]
        delta_f = design_factor @ z
        delta_cf = cf_design_factor @ z
        eta_f = eta_factual[:, None] + delta_f
        eta_cf = eta_counterfactual[:, None] + delta_cf
        finite = np.all(np.isfinite(eta_f) & np.isfinite(eta_cf), axis=0)
        finite &= np.max(eta_f, axis=0) < 700.0
        finite &= np.max(eta_cf, axis=0) < 700.0
        if not np.any(finite):
            continue
        local_indices = np.flatnonzero(finite)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            factual_sum = np.exp(eta_f[:, finite]).sum(axis=0)
            counterfactual_sum = np.exp(eta_cf[:, finite]).sum(axis=0)
            contrast = factual_sum - counterfactual_sum
        local_valid = np.isfinite(contrast)
        target_indices = start + local_indices[local_valid]
        values[target_indices] = contrast[local_valid]
        valid[target_indices] = True
    return values, valid


def _blank_interval_row(base: dict[str, Any], covariance_method: str, interval_method: str) -> dict[str, Any]:
    return {
        **base,
        "covariance_method": covariance_method,
        "interval_method": interval_method,
        "n_draws": PRIMARY_DRAWS if interval_method != "normal_wald" else 0,
        "interval_lo": np.nan,
        "interval_hi": np.nan,
        "interval_width": np.nan,
        "interval_asymmetry_upper_minus_lower": np.nan,
        "interval_center_shift_from_A_hat": np.nan,
        "interval_valid": False,
        "covered": False,
        "covered_planned": False,
    }


def _process_one(task: tuple[str, int]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("MVN replay process was not initialized.")
    started = time.perf_counter()
    legacy = _WORKER_CONTEXT["legacy"]
    case_state = _WORKER_CONTEXT["case_state"]
    scenario, replicate = task
    engine = case_state["engine"]
    original_fit_function = engine.fit_poisson_irls
    captured: dict[str, Any] = {}

    def capture_fit(*args, **kwargs):
        fit = original_fit_function(*args, **kwargs)
        captured["fit"] = fit
        return fit

    engine.fit_poisson_irls = capture_fit
    try:
        replay = legacy._one_replay((scenario, replicate))
    finally:
        engine.fit_poisson_irls = original_fit_function

    config = case_state["config"]
    sim = config["simulation"]
    scenario_id = int(case_state["scenario_ids"][scenario])
    draw_seed = [
        int(sim["base_seed"]),
        int(sim["mode_ids"]["formal"]),
        scenario_id,
        int(replicate),
        DRAW_SEED_TAG,
    ]
    base = {
        "case": case_state["case"],
        "scenario": scenario,
        "replicate": int(replicate),
        "seed_entropy": replay["seed_entropy"],
        "outcome_sha256": replay["outcome_sha256"],
        "draw_seed_entropy": json.dumps(draw_seed, separators=(",", ":")),
        "converged": bool(replay["converged"]),
        "fit_failure_reason": str(replay["failure_reason"]),
        "iterations": replay["iterations"],
        "A_true": float(replay["A_true"]),
        "A_hat": replay["A_hat"],
        "A_hat_original": replay["A_hat_original"],
        "se_model_replay": replay["se_model_replay"],
        "se_model_original": replay["se_model_original"],
        "relative_A_hat_difference": replay["relative_A_hat_difference"],
        "relative_se_model_difference": replay["relative_se_model_difference"],
        "match_A_hat_1e_8": bool(replay["match_A_hat_1e_8"]),
        "match_se_model_1e_8": bool(replay["match_se_model_1e_8"]),
        "phi_hat": replay["phi_hat"],
        "replay_elapsed_seconds": np.nan,
    }
    fit = captured.get("fit")
    p = int(case_state["design"]["X"].shape[1])
    factors = {
        "beta": np.full(p, np.nan, dtype=float),
        "factor_quasi_pearson": np.full((p, p), np.nan, dtype=float),
        "factor_hac28": np.full((p, p), np.nan, dtype=float),
    }
    stability_rows: list[dict[str, Any]] = []
    if not replay["converged"] or fit is None:
        rows = [
            _blank_interval_row(
                {
                    **base,
                    "covariance_minimum_eigenvalue": np.nan,
                    "covariance_psd_tolerance": np.nan,
                    "covariance_psd_valid": False,
                },
                covariance_name,
                interval_name,
            )
            for covariance_name in ("quasi_pearson", "hac28")
            for interval_name in ("normal_wald", "linear_gaussian_mc", "exact_nonlinear_mc")
        ]
        base["replay_elapsed_seconds"] = time.perf_counter() - started
        return {"rows": rows, "stability": stability_rows, "factors": factors, "hash": replay["outcome_sha256"]}

    X = np.asarray(case_state["design"]["X"], dtype=float)
    Xcf = np.asarray(case_state["design"]["Xcf"], dtype=float)
    offset = np.asarray(case_state["design"]["offset"], dtype=float)
    beta = np.asarray(fit["beta"], dtype=float)
    mu = np.asarray(fit["mu"], dtype=float)
    mu_cf = np.asarray(fit["mu_cf"], dtype=float)
    eta_factual = offset + X @ beta
    eta_counterfactual = offset + Xcf @ beta
    a_hat = float(np.exp(eta_factual).sum() - np.exp(eta_counterfactual).sum())
    if not math.isfinite(a_hat):
        raise FloatingPointError("Per-replicate fitted attribution was non-finite.")
    gradient = X.T @ mu - Xcf.T @ mu_cf
    hessian = X.T @ (mu[:, None] * X)
    bread = solve(hessian, np.eye(p), assume_a="pos", check_finite=False)
    bread = (bread + bread.T) / 2.0
    phi_hat = float(replay["phi_hat"])
    covariance_matrices = {
        "quasi_pearson": phi_hat * bread,
        "hac28": np.asarray(fit["beta_covariance_last_hac"], dtype=float),
    }
    factor_info: dict[str, tuple[np.ndarray | None, float, float, bool]] = {
        name: _stable_factor(value) for name, value in covariance_matrices.items()
    }
    factors["beta"] = beta.copy()

    is_stability = int(replicate) < STABILITY_REPLICATES
    maximum_draws = max(STABILITY_DRAWS) if is_stability else PRIMARY_DRAWS
    rng = np.random.default_rng(np.random.SeedSequence(draw_seed))
    positive = rng.standard_normal((p, maximum_draws // 2))
    z_draws = np.concatenate((positive, -positive), axis=1)
    draw_sizes = [PRIMARY_DRAWS, *STABILITY_DRAWS] if is_stability else [PRIMARY_DRAWS]
    zcrit = float(case_state["zcrit"])
    truth = float(replay["A_true"])
    all_rows: list[dict[str, Any]] = []

    for covariance_name, covariance in covariance_matrices.items():
        factor, min_eigenvalue, psd_tolerance, covariance_valid = factor_info[covariance_name]
        if factor is None:
            design_factor = None
            cf_design_factor = None
            linear_values = None
            exact_values = None
            exact_valid = None
            factor_for_file = np.full((p, p), np.nan, dtype=float)
        else:
            factor_for_file = factor
            design_factor = X @ factor
            cf_design_factor = Xcf @ factor
            gradient_factor = gradient @ factor
            linear_values = a_hat + gradient_factor @ z_draws
            exact_values, exact_valid = _exact_nonlinear_draws(
                design_factor,
                cf_design_factor,
                eta_factual,
                eta_counterfactual,
                z_draws,
            )
        factors["factor_quasi_pearson" if covariance_name == "quasi_pearson" else "factor_hac28"] = factor_for_file
        covariance_details = {
            "covariance_minimum_eigenvalue": min_eigenvalue,
            "covariance_psd_tolerance": psd_tolerance,
            "covariance_psd_valid": bool(covariance_valid),
        }
        common = {**base, **covariance_details}
        if covariance_valid:
            assert factor is not None
            normal_sd = float(np.linalg.norm(gradient @ factor))
            normal_row = _blank_interval_row(common, covariance_name, "normal_wald")
            normal_row.update({
                "n_draws": 0,
                "interval_lo": a_hat - zcrit * normal_sd,
                "interval_hi": a_hat + zcrit * normal_sd,
                "interval_width": 2.0 * zcrit * normal_sd,
                "interval_asymmetry_upper_minus_lower": 0.0,
                "interval_center_shift_from_A_hat": 0.0,
                "interval_valid": True,
            })
            normal_row["covered"] = bool(normal_row["interval_lo"] <= truth <= normal_row["interval_hi"])
            normal_row["covered_planned"] = normal_row["covered"]
            all_rows.append(normal_row)
        else:
            all_rows.append(_blank_interval_row(common, covariance_name, "normal_wald"))

        for interval_name, draw_values in (
            ("linear_gaussian_mc", linear_values),
            ("exact_nonlinear_mc", exact_values),
        ):
            for draw_count in draw_sizes:
                index = _antithetic_indices(draw_count, maximum_draws)
                selected = None if draw_values is None else np.asarray(draw_values)[index]
                if interval_name == "exact_nonlinear_mc":
                    selected_valid = (
                        np.zeros(draw_count, dtype=bool)
                        if exact_valid is None else np.asarray(exact_valid)[index]
                    )
                    sample_valid = bool(np.all(selected_valid))
                    if selected is not None:
                        sample_valid = sample_valid and bool(np.all(np.isfinite(selected)))
                else:
                    sample_valid = selected is not None and bool(np.all(np.isfinite(selected)))
                interval_valid = bool(covariance_valid and sample_valid)
                if interval_valid:
                    lo, hi = (float(x) for x in np.quantile(selected, [0.05, 0.95]))
                    width = float(hi - lo)
                    covered = bool(lo <= truth <= hi)
                else:
                    lo = hi = width = np.nan
                    covered = False
                if draw_count == PRIMARY_DRAWS:
                    row = _blank_interval_row(common, covariance_name, interval_name)
                    row.update({
                        "n_draws": draw_count,
                        "interval_lo": lo,
                        "interval_hi": hi,
                        "interval_width": width,
                        "interval_asymmetry_upper_minus_lower": (
                            float((hi - a_hat) - (a_hat - lo)) if interval_valid else np.nan
                        ),
                        "interval_center_shift_from_A_hat": (
                            float((hi + lo) / 2.0 - a_hat) if interval_valid else np.nan
                        ),
                        "interval_valid": interval_valid,
                        "covered": covered,
                        "covered_planned": bool(interval_valid and covered),
                    })
                    all_rows.append(row)
                if is_stability:
                    stability_rows.append({
                        "case": case_state["case"],
                        "scenario": scenario,
                        "replicate": int(replicate),
                        "covariance_method": covariance_name,
                        "interval_method": interval_name,
                        "n_draws": draw_count,
                        "interval_valid": interval_valid,
                        "interval_lo": lo,
                        "interval_hi": hi,
                        "interval_width": width,
                        "covered": bool(interval_valid and covered),
                    })

    if is_stability:
        stable = pd.DataFrame(stability_rows)
        for (covariance_name, interval_name), positions in stable.groupby(
            ["covariance_method", "interval_method"], sort=False
        ).groups.items():
            subset = stable.loc[positions]
            primary = subset.loc[subset["n_draws"].eq(PRIMARY_DRAWS)]
            if primary.empty or not bool(primary.iloc[0]["interval_valid"]):
                stable.loc[positions, "lower_change_vs_2048"] = np.nan
                stable.loc[positions, "upper_change_vs_2048"] = np.nan
                stable.loc[positions, "width_change_vs_2048"] = np.nan
                continue
            base_lo = float(primary.iloc[0]["interval_lo"])
            base_hi = float(primary.iloc[0]["interval_hi"])
            base_width = float(primary.iloc[0]["interval_width"])
            for idx in positions:
                if not bool(stable.at[idx, "interval_valid"]):
                    stable.at[idx, "lower_change_vs_2048"] = np.nan
                    stable.at[idx, "upper_change_vs_2048"] = np.nan
                    stable.at[idx, "width_change_vs_2048"] = np.nan
                else:
                    stable.at[idx, "lower_change_vs_2048"] = float(stable.at[idx, "interval_lo"]) - base_lo
                    stable.at[idx, "upper_change_vs_2048"] = float(stable.at[idx, "interval_hi"]) - base_hi
                    stable.at[idx, "width_change_vs_2048"] = float(stable.at[idx, "interval_width"]) - base_width
        stability_rows = stable.to_dict(orient="records")

    base["replay_elapsed_seconds"] = time.perf_counter() - started
    for row in all_rows:
        row["replay_elapsed_seconds"] = base["replay_elapsed_seconds"]
    return {
        "rows": all_rows,
        "stability": stability_rows,
        "factors": factors,
        "hash": replay["outcome_sha256"],
        "elapsed_seconds": base["replay_elapsed_seconds"],
        "replay_ok": bool(replay["converged"] and replay["match_A_hat_1e_8"] and replay["match_se_model_1e_8"]),
    }


def _summaries(replications: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    paired_rows = []
    for (case, scenario, covariance, method), group in replications.groupby(
        ["case", "scenario", "covariance_method", "interval_method"], sort=False
    ):
        n_planned = int(len(group))
        valid = group.loc[group["interval_valid"].astype(bool)]
        coverage = float(group["covered_planned"].astype(bool).mean()) if n_planned else np.nan
        n_covered = int(group["covered_planned"].astype(bool).sum())
        widths = pd.to_numeric(valid["interval_width"], errors="coerce").to_numpy(dtype=float)
        widths = widths[np.isfinite(widths)]
        asymmetries = pd.to_numeric(
            valid["interval_asymmetry_upper_minus_lower"], errors="coerce"
        ).to_numpy(dtype=float)
        asymmetries = asymmetries[np.isfinite(asymmetries)]
        summary_rows.append({
            "case": case,
            "scenario": scenario,
            "covariance_method": covariance,
            "interval_method": method,
            "n_planned": n_planned,
            "n_valid_intervals": int(len(valid)),
            "n_invalid_intervals_counted_uncovered": int(n_planned - len(valid)),
            "n_covered_planned": n_covered,
            "coverage_planned": coverage,
            "coverage_mcse_planned": math.sqrt(coverage * (1.0 - coverage) / n_planned) if n_planned else np.nan,
            "coverage_valid_only": (
                float(valid["covered"].astype(bool).mean()) if len(valid) else np.nan
            ),
            "mean_width_valid_only": (
                float(np.mean(widths)) if len(widths) else np.nan
            ),
            "mcse_mean_width_valid_only": (
                float(np.std(widths, ddof=1) / math.sqrt(len(widths))) if len(widths) > 1 else np.nan
            ),
            "mean_interval_asymmetry_upper_minus_lower": (
                float(np.mean(asymmetries)) if len(asymmetries) else np.nan
            ),
            "mcse_mean_interval_asymmetry_valid_only": (
                float(np.std(asymmetries, ddof=1) / math.sqrt(len(asymmetries)))
                if len(asymmetries) > 1 else np.nan
            ),
            "mean_interval_center_shift_from_A_hat": (
                float(pd.to_numeric(valid["interval_center_shift_from_A_hat"], errors="coerce").mean())
                if len(valid) else np.nan
            ),
            "n_psd_covariances": int(
                group[["replicate", "covariance_psd_valid"]]
                .drop_duplicates("replicate")["covariance_psd_valid"].astype(bool).sum()
            ),
            "n_nonpsd_covariances": int(
                (~group[["replicate", "covariance_psd_valid"]]
                 .drop_duplicates("replicate")["covariance_psd_valid"].astype(bool)).sum()
            ),
        })
    for (case, scenario, method), group in replications.groupby(
        ["case", "scenario", "interval_method"], sort=False
    ):
        q = group.loc[group["covariance_method"].eq("quasi_pearson")].set_index("replicate")
        h = group.loc[group["covariance_method"].eq("hac28")].set_index("replicate")
        common_ids = sorted(set(q.index) & set(h.index))
        q = q.loc[common_ids]
        h = h.loc[common_ids]
        q_cover = q["covered_planned"].astype(float).to_numpy()
        h_cover = h["covered_planned"].astype(float).to_numpy()
        cover_diff = q_cover - h_cover
        q_valid = q["interval_valid"].astype(bool).to_numpy()
        h_valid = h["interval_valid"].astype(bool).to_numpy()
        width_valid = q_valid & h_valid
        width_diff = (
            pd.to_numeric(q["interval_width"], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(h["interval_width"], errors="coerce").to_numpy(dtype=float)
        )
        width_diff = width_diff[width_valid & np.isfinite(width_diff)]
        n_planned = len(common_ids)
        paired_rows.append({
            "case": case,
            "scenario": scenario,
            "interval_method": method,
            "method_1": "quasi_pearson",
            "method_2": "hac28",
            "n_planned_pairs": n_planned,
            "mean_paired_coverage_difference_quasi_minus_hac28": (
                float(np.mean(cover_diff)) if n_planned else np.nan
            ),
            "mcse_paired_coverage_difference": (
                float(np.std(cover_diff, ddof=1) / math.sqrt(n_planned)) if n_planned > 1 else np.nan
            ),
            "n_jointly_valid_width_pairs": int(len(width_diff)),
            "mean_paired_width_difference_quasi_minus_hac28": (
                float(np.mean(width_diff)) if len(width_diff) else np.nan
            ),
            "mcse_paired_width_difference": (
                float(np.std(width_diff, ddof=1) / math.sqrt(len(width_diff))) if len(width_diff) > 1 else np.nan
            ),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def _inference_method_paired(replications: pd.DataFrame) -> pd.DataFrame:
    """Paired interval-method contrasts within the same covariance and Y draw."""
    rows = []
    comparisons = (
        ("exact_nonlinear_mc", "normal_wald"),
        ("exact_nonlinear_mc", "linear_gaussian_mc"),
    )
    for (case, scenario, covariance), group in replications.groupby(
        ["case", "scenario", "covariance_method"], sort=False
    ):
        by_method = {
            method: frame.set_index("replicate")
            for method, frame in group.groupby("interval_method", sort=False)
        }
        for method_1, method_2 in comparisons:
            if method_1 not in by_method or method_2 not in by_method:
                continue
            first = by_method[method_1]
            second = by_method[method_2]
            replicate_ids = sorted(set(first.index) & set(second.index))
            first = first.loc[replicate_ids]
            second = second.loc[replicate_ids]
            n_planned = len(replicate_ids)
            coverage_difference = (
                first["covered_planned"].astype(float).to_numpy()
                - second["covered_planned"].astype(float).to_numpy()
            )
            first_valid = first["interval_valid"].astype(bool).to_numpy()
            second_valid = second["interval_valid"].astype(bool).to_numpy()
            joint_valid = first_valid & second_valid
            width_difference = (
                pd.to_numeric(first["interval_width"], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(second["interval_width"], errors="coerce").to_numpy(dtype=float)
            )
            width_difference = width_difference[joint_valid & np.isfinite(width_difference)]
            rows.append({
                "case": case,
                "scenario": scenario,
                "covariance_method": covariance,
                "method_1": method_1,
                "method_2": method_2,
                "n_planned_pairs": n_planned,
                "mean_paired_coverage_difference_method_1_minus_method_2": (
                    float(np.mean(coverage_difference)) if n_planned else np.nan
                ),
                "mcse_paired_coverage_difference": (
                    float(np.std(coverage_difference, ddof=1) / math.sqrt(n_planned))
                    if n_planned > 1 else np.nan
                ),
                "n_jointly_valid_width_pairs": int(len(width_difference)),
                "mean_paired_width_difference_method_1_minus_method_2": (
                    float(np.mean(width_difference)) if len(width_difference) else np.nan
                ),
                "mcse_paired_width_difference": (
                    float(np.std(width_difference, ddof=1) / math.sqrt(len(width_difference)))
                    if len(width_difference) > 1 else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _save_factors(path: Path, factor_records: list[dict[str, Any]], compressed: bool = True) -> None:
    ordered = sorted(factor_records, key=lambda x: (x["scenario_order"], x["replicate"]))
    if not ordered:
        return
    payload = {
        "scenario": np.asarray([x["scenario"] for x in ordered], dtype="U64"),
        "replicate": np.asarray([x["replicate"] for x in ordered], dtype=np.int64),
        "outcome_sha256": np.asarray([x["outcome_sha256"] for x in ordered], dtype="U64"),
        "beta": np.stack([x["beta"] for x in ordered]),
        "factor_quasi_pearson": np.stack([x["factor_quasi_pearson"] for x in ordered]),
        "factor_hac28": np.stack([x["factor_hac28"] for x in ordered]),
        "min_eigenvalue_quasi_pearson": np.asarray([x["min_eigenvalue_quasi_pearson"] for x in ordered]),
        "min_eigenvalue_hac28": np.asarray([x["min_eigenvalue_hac28"] for x in ordered]),
        "psd_tolerance_quasi_pearson": np.asarray([x["psd_tolerance_quasi_pearson"] for x in ordered]),
        "psd_tolerance_hac28": np.asarray([x["psd_tolerance_hac28"] for x in ordered]),
        "valid_quasi_pearson": np.asarray([x["valid_quasi_pearson"] for x in ordered], dtype=bool),
        "valid_hac28": np.asarray([x["valid_hac28"] for x in ordered], dtype=bool),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **payload)
        else:
            np.savez(handle, **payload)
    temp.replace(path)


def _load_factor_records(source: Any, scenario_order: dict[str, int]) -> list[dict[str, Any]]:
    records = []
    with np.load(source, allow_pickle=False) as bundle:
        scenarios = bundle["scenario"]
        replicates = bundle["replicate"]
        outcome_hashes = bundle["outcome_sha256"]
        betas = bundle["beta"]
        factors_quasi = bundle["factor_quasi_pearson"]
        factors_hac28 = bundle["factor_hac28"]
        min_eigenvalues_quasi = bundle["min_eigenvalue_quasi_pearson"]
        min_eigenvalues_hac28 = bundle["min_eigenvalue_hac28"]
        tolerances_quasi = bundle["psd_tolerance_quasi_pearson"]
        tolerances_hac28 = bundle["psd_tolerance_hac28"]
        valid_quasi = bundle["valid_quasi_pearson"]
        valid_hac28 = bundle["valid_hac28"]
        count = len(replicates)
        for index in range(count):
            scenario = str(scenarios[index])
            records.append({
                "scenario": scenario,
                "replicate": int(replicates[index]),
                "scenario_order": int(scenario_order[scenario]),
                "outcome_sha256": str(outcome_hashes[index]),
                "beta": np.asarray(betas[index], dtype=float).copy(),
                "factor_quasi_pearson": np.asarray(factors_quasi[index], dtype=float).copy(),
                "factor_hac28": np.asarray(factors_hac28[index], dtype=float).copy(),
                "min_eigenvalue_quasi_pearson": float(min_eigenvalues_quasi[index]),
                "min_eigenvalue_hac28": float(min_eigenvalues_hac28[index]),
                "psd_tolerance_quasi_pearson": float(tolerances_quasi[index]),
                "psd_tolerance_hac28": float(tolerances_hac28[index]),
                "valid_quasi_pearson": bool(valid_quasi[index]),
                "valid_hac28": bool(valid_hac28[index]),
            })
    return records


def _checkpoint_paths(case: str) -> dict[str, Path]:
    prefix = NEW_ROOT / "results" / f"mvn_formal_{case}_checkpoint"
    return {"bundle": prefix.with_suffix(".zip")}


def _save_checkpoint(
    paths: dict[str, Path],
    results: list[dict[str, Any]],
    stability_records: list[dict[str, Any]],
    factor_records: list[dict[str, Any]],
) -> None:
    if not results or not factor_records:
        return
    bundle_path = paths["bundle"]
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    factor_temp = bundle_path.with_name(bundle_path.stem + ".factors.tmp.npz")
    archive_temp = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    _save_factors(factor_temp, factor_records, compressed=False)
    rep_bytes = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
    stability_frame = pd.DataFrame(stability_records)
    if stability_frame.empty:
        stability_frame = pd.DataFrame(columns=[
            "case", "scenario", "replicate", "covariance_method", "interval_method",
            "n_draws", "interval_valid", "interval_lo", "interval_hi", "interval_width", "covered",
        ])
    stability_bytes = stability_frame.to_csv(index=False).encode("utf-8")
    manifest = {
        "created_utc": _utc_now(),
        "completed_replications": len(factor_records),
        "replication_rows": len(results),
        "stability_rows": len(stability_records),
    }
    try:
        with zipfile.ZipFile(archive_temp, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("checkpoint.json", json.dumps(manifest, indent=2, ensure_ascii=False), compress_type=zipfile.ZIP_STORED)
            archive.writestr("replications.csv", rep_bytes, compress_type=zipfile.ZIP_STORED)
            archive.writestr("stability.csv", stability_bytes, compress_type=zipfile.ZIP_STORED)
            archive.write(factor_temp, "factors.npz", compress_type=zipfile.ZIP_STORED)
        archive_temp.replace(bundle_path)
    finally:
        if factor_temp.exists():
            factor_temp.unlink()
        if archive_temp.exists():
            archive_temp.unlink()


def _load_checkpoint(
    paths: dict[str, Path],
    scenarios: list[str],
    reps_per_scenario: int,
    scenario_order: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, int]]]:
    bundle_path = paths["bundle"]
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Formal checkpoint is missing: {bundle_path}")
    with zipfile.ZipFile(bundle_path, "r") as archive:
        frame = pd.read_csv(io.BytesIO(archive.read("replications.csv")))
        stability_frame = pd.read_csv(io.BytesIO(archive.read("stability.csv")))
        factor_records = _load_factor_records(
            io.BytesIO(archive.read("factors.npz")), scenario_order
        )
    expected_methods = {
        (covariance, method)
        for covariance in ("quasi_pearson", "hac28")
        for method in ("normal_wald", "linear_gaussian_mc", "exact_nonlinear_mc")
    }
    done: set[tuple[str, int]] = set()
    for key, group in frame.groupby(["scenario", "replicate"], sort=False):
        scenario, replicate = str(key[0]), int(key[1])
        if scenario not in scenarios or not 0 <= replicate < reps_per_scenario:
            raise ValueError(f"Checkpoint contains an out-of-plan task: {key}")
        observed = set(zip(group["covariance_method"].astype(str), group["interval_method"].astype(str)))
        if len(group) != 6 or observed != expected_methods:
            raise ValueError(f"Checkpoint task is incomplete: {key}")
        done.add((scenario, replicate))
    factor_keys = {(record["scenario"], int(record["replicate"])) for record in factor_records}
    if factor_keys != done:
        raise ValueError("Checkpoint factor keys do not match completed replication rows.")
    result_records = frame.to_dict(orient="records")
    stability_records = stability_frame.to_dict(orient="records")
    return result_records, stability_records, factor_records, done


def _factor_record(result: dict[str, Any], scenario: str, replicate: int, scenario_order: int) -> dict[str, Any]:
    rows = result["rows"]
    by_covariance = {}
    for name in ("quasi_pearson", "hac28"):
        matches = [row for row in rows if row["covariance_method"] == name]
        by_covariance[name] = matches[0] if matches else {}
    factors = result["factors"]
    return {
        "scenario": scenario,
        "replicate": int(replicate),
        "scenario_order": int(scenario_order),
        "outcome_sha256": result["hash"],
        "beta": factors["beta"],
        "factor_quasi_pearson": factors["factor_quasi_pearson"],
        "factor_hac28": factors["factor_hac28"],
        "min_eigenvalue_quasi_pearson": by_covariance["quasi_pearson"].get("covariance_minimum_eigenvalue", np.nan),
        "min_eigenvalue_hac28": by_covariance["hac28"].get("covariance_minimum_eigenvalue", np.nan),
        "psd_tolerance_quasi_pearson": by_covariance["quasi_pearson"].get("covariance_psd_tolerance", np.nan),
        "psd_tolerance_hac28": by_covariance["hac28"].get("covariance_psd_tolerance", np.nan),
        "valid_quasi_pearson": bool(by_covariance["quasi_pearson"].get("covariance_psd_valid", False)),
        "valid_hac28": bool(by_covariance["hac28"].get("covariance_psd_valid", False)),
    }


def _case_repro(
    case: str,
    mode: str,
    started: float,
    replications_per_scenario: int,
    workers: int,
    replications: pd.DataFrame,
    stability: pd.DataFrame,
    case_state: dict[str, Any],
    outputs: list[str],
) -> dict[str, Any]:
    formal_repro = case_state["formal_repro"]
    outcome_table = replications.drop_duplicates(["scenario", "replicate"])
    expected = len(case_state["scenario_ids"]) * replications_per_scenario
    replay_checks = bool(
        len(outcome_table) == expected
        and outcome_table["converged"].astype(bool).all()
        and outcome_table["match_A_hat_1e_8"].astype(bool).all()
        and outcome_table["match_se_model_1e_8"].astype(bool).all()
    )
    return {
        "case": case,
        "mode": mode,
        "started_utc": _utc_now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "project_root": str(NEW_ROOT),
        "frozen_project_root_read_only": str(OLD_ROOT),
        "python": sys.version,
        "python_executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "runtime_prefix": sys.prefix,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "workers": min(int(workers), 4),
        "blas_threads_per_worker": 1,
        "planned_replications_per_scenario": int(replications_per_scenario),
        "planned_outcomes": int(expected),
        "primary_draws": PRIMARY_DRAWS,
        "antithetic_draws": True,
        "stability_draws": list(STABILITY_DRAWS),
        "stability_replicates_per_scenario": STABILITY_REPLICATES,
        "psd_relative_tolerance": PSD_RELATIVE_TOLERANCE,
        "input_sha256": formal_repro["input"]["sha256"],
        "approved_config_sha256": formal_repro["config_sha256"],
        "engine_code_sha256": formal_repro["code_sha256"],
        "old_real_panel_audit_sha256": _sha256(case_state["root"] / "code" / "real_panel_audit.py"),
        "quasi_baseline_sha256": _sha256(OLD_ROOT / "code" / "quasi_baseline.py"),
        "mvn_script_sha256": _sha256(Path(__file__)),
        "formal_reproducibility_sha256": _sha256(case_state["root"] / "results" / "real_formal_reproducibility.json"),
        "seed_replay": {
            "outcome_seed_rule": "legacy SeedSequence([base_seed, formal_mode_id, scenario_seed_id, replicate_index])",
            "mvn_draw_seed_rule": "SeedSequence([base_seed, formal_mode_id, scenario_seed_id, replicate_index, 78111]); same Z shared by Quasi and HAC28",
            "unique_outcome_hashes": int(outcome_table["outcome_sha256"].nunique()),
            "expected_unique_outcome_hashes": int(expected),
            "all_replay_A_hat_and_model_SE_match_within_relative_1e_8": replay_checks,
            "max_relative_A_hat_difference": float(pd.to_numeric(outcome_table["relative_A_hat_difference"], errors="coerce").max()),
            "max_relative_model_SE_difference": float(pd.to_numeric(outcome_table["relative_se_model_difference"], errors="coerce").max()),
        },
        "covariances": {
            "quasi_pearson": "phi_hat * inverse(X' diag(mu_hat) X); phi_hat = Pearson sum((y-mu)^2/mu)/(n-p), no floor",
            "hac28": "per-replicate fit_poisson_irls beta_covariance_last_hac from Bartlett HAC bandwidth 28",
            "psd_rule": "symmetric eigendecomposition; accept min_eigenvalue >= -1e-10*max(1,max_abs_eigenvalue); clip only tolerated negative eigenvalues to zero for factorization",
        },
        "intervals": {
            "normal_wald": "fitted A_hat +/- approved normal critical value times sqrt(g' V g)",
            "linear_gaussian_mc": "5th/95th empirical percentiles of A_hat + g'(beta_draw-beta_hat)",
            "exact_nonlinear_mc": "5th/95th empirical percentiles of sum(exp(offset+X beta_draw))-sum(exp(offset+Xcf beta_draw))",
            "invalid_covariance_or_interval": "planned uncovered; no draws from materially indefinite covariance",
        },
        "all_seeded_replay_checks_pass": replay_checks,
        "outputs": outputs,
    }


def resummarize_existing(mode: str, case_choice: str) -> dict[str, Any]:
    """Refresh summary tables from saved replicate outputs without refitting Y."""
    if mode not in {"pilot", "formal"}:
        raise ValueError("mode must be pilot or formal")
    cases = ("state", "city") if case_choice == "both" else (case_choice,)
    refreshed = []
    for case in cases:
        if case not in {"state", "city"}:
            raise ValueError("case must be state, city, or both")
        name = f"mvn_{mode}_{case}"
        rep_path = NEW_ROOT / "results" / f"{name}_replications.csv"
        if not rep_path.is_file():
            raise FileNotFoundError(f"Saved replicate table is missing: {rep_path}")
        replications = pd.read_csv(rep_path)
        if "interval_asymmetry_upper_minus_lower" not in replications.columns:
            interval_valid = replications["interval_valid"].astype(bool)
            lower = pd.to_numeric(replications["interval_lo"], errors="coerce")
            upper = pd.to_numeric(replications["interval_hi"], errors="coerce")
            estimate = pd.to_numeric(replications["A_hat"], errors="coerce")
            replications["interval_asymmetry_upper_minus_lower"] = np.where(
                interval_valid, (upper - estimate) - (estimate - lower), np.nan
            )
            replications["interval_center_shift_from_A_hat"] = np.where(
                interval_valid, (upper + lower) / 2.0 - estimate, np.nan
            )
            _write_csv(rep_path, replications)
        summary, covariance_paired = _summaries(replications)
        method_paired = _inference_method_paired(replications)
        summary_path = NEW_ROOT / "tables" / f"{name}_summary.csv"
        paired_path = NEW_ROOT / "tables" / f"{name}_paired.csv"
        method_paired_path = NEW_ROOT / "tables" / f"{name}_method_paired.csv"
        _write_csv(summary_path, summary)
        _write_csv(paired_path, covariance_paired)
        _write_csv(method_paired_path, method_paired)
        repro_path = NEW_ROOT / "results" / f"{name}_reproducibility.json"
        if repro_path.is_file():
            repro = json.loads(repro_path.read_text(encoding="utf-8"))
            outputs = set(repro.get("outputs", []))
            outputs.add(str(method_paired_path.relative_to(NEW_ROOT)))
            repro["outputs"] = sorted(outputs)
            _write_json(repro_path, repro)
        refreshed.append({
            "case": case,
            "n_replication_rows": int(len(replications)),
            "summary": str(summary_path),
            "covariance_paired": str(paired_path),
            "interval_method_paired": str(method_paired_path),
        })
    return {"mode": mode, "resummarized_without_replay": True, "cases": refreshed}


def run(mode: str, case_choice: str, pilot_reps: int, workers: int, resume: bool = False) -> dict[str, Any]:
    if mode not in {"pilot", "formal"}:
        raise ValueError("mode must be pilot or formal")
    if mode == "pilot" and not 1 <= pilot_reps <= PILOT_REPLICATIONS_PER_SCENARIO_MAX:
        raise ValueError(f"pilot_reps must be between 1 and {PILOT_REPLICATIONS_PER_SCENARIO_MAX}")
    if resume and mode != "formal":
        raise ValueError("resume is only available for formal runs")
    if workers < 1:
        raise ValueError("workers must be at least one")
    cases = ("state", "city") if case_choice == "both" else (case_choice,)
    if any(case not in {"state", "city"} for case in cases):
        raise ValueError("case must be state, city, or both")
    workers = min(workers, 4)
    status_path = NEW_ROOT / "logs" / f"mvn_{mode}_status.json"
    prior_status: dict[str, Any] = {}
    if mode == "formal" and resume and status_path.is_file():
        try:
            prior_status = json.loads(status_path.read_text(encoding="utf-8-sig"))
        except Exception:
            prior_status = {}
    started_all = time.perf_counter()
    progress = {
        "status": "running",
        "mode": mode,
        "pid": os.getpid(),
        "started_utc": _utc_now(),
        "new_root": str(NEW_ROOT),
        "old_root_read_only": str(OLD_ROOT),
        "cases": list(cases),
        "workers": workers,
        "blas_threads_per_worker": 1,
        "completed": 0,
        "planned": 0,
    }
    if prior_status:
        progress["resumed_from_pid"] = prior_status.get("pid")
        progress["previous_elapsed_seconds"] = prior_status.get("elapsed_seconds")
        progress["previous_reported_completed"] = prior_status.get("case_completed")
        progress["previous_checkpoint_completed"] = prior_status.get("checkpoint_completed_replications")
        progress["previous_uncheckpointed_delta"] = prior_status.get("reported_uncheckpointed_delta")
        progress["previous_checkpoint_sha256"] = prior_status.get("checkpoint_sha256")
    _write_json(status_path, progress)
    all_outcome_hashes: list[str] = []
    case_checks: list[dict[str, Any]] = []

    try:
        for case in cases:
            case_started = time.perf_counter()
            legacy = _legacy_module()
            case_state = legacy._prepare_case(case)
            original_rows = _validate_formal_rows(case_state)
            scenarios = list(case_state["scenario_means"])
            reps_per_scenario = (
                pilot_reps if mode == "pilot" else FORMAL_REPLICATIONS_PER_SCENARIO
            )
            tasks = [
                (scenario, replicate)
                for scenario in scenarios
                for replicate in range(reps_per_scenario)
            ]
            results: list[dict[str, Any]] = []
            stability_records: list[dict[str, Any]] = []
            factor_records: list[dict[str, Any]] = []
            scenario_order = {scenario: index for index, scenario in enumerate(scenarios)}
            checkpoint_paths = _checkpoint_paths(case) if mode == "formal" else None
            done_tasks: set[tuple[str, int]] = set()
            if mode == "formal":
                assert checkpoint_paths is not None
                checkpoint_exists = any(path.exists() for path in checkpoint_paths.values())
                final_replications_path = NEW_ROOT / "results" / f"mvn_formal_{case}_replications.csv"
                if resume:
                    if checkpoint_exists:
                        results, stability_records, factor_records, done_tasks = _load_checkpoint(
                            checkpoint_paths, scenarios, reps_per_scenario, scenario_order
                        )
                        completed_hashes = (
                            pd.DataFrame(results)
                            .drop_duplicates(["scenario", "replicate"])["outcome_sha256"]
                            .astype(str)
                            .tolist()
                        )
                        all_outcome_hashes.extend(completed_hashes)
                    elif final_replications_path.exists():
                        raise FileExistsError(
                            f"Final formal outputs already exist for {case}; omit this completed case when resuming."
                        )
                else:
                    if checkpoint_exists:
                        raise FileExistsError(
                            f"A formal checkpoint exists for {case}; resume it with --resume instead of overwriting."
                        )
                    if final_replications_path.exists():
                        raise FileExistsError(
                            f"Formal outputs already exist for {case}; use a fresh root or inspect existing results."
                        )
            elif resume:
                raise ValueError("Pilot runs cannot be resumed.")
            tasks_pending = [task for task in tasks if task not in done_tasks]
            completed_before = len(done_tasks)
            progress["completed"] += completed_before
            progress.update({
                "current_case": case,
                "case_completed": completed_before,
                "case_planned": len(tasks),
                "planned": progress["completed"] + len(tasks_pending),
            })
            if checkpoint_paths is not None:
                progress["checkpoint_files"] = {key: str(path) for key, path in checkpoint_paths.items()}
                progress["resume_command"] = (
                    f'"{EXPECTED_PYTHON}" -X utf8 "{Path(__file__)}" --mode formal --case {case} '
                    f"--workers 4 --formal-approved --resume"
                )
            _write_json(status_path, progress)
            finished = 0
            if tasks_pending:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_process,
                    initargs=(case,),
                ) as pool:
                    futures = {pool.submit(_process_one, task): task for task in tasks_pending}
                    for future in concurrent.futures.as_completed(futures):
                        task = futures[future]
                        try:
                            result = future.result()
                        except Exception:
                            if checkpoint_paths is not None:
                                _save_checkpoint(checkpoint_paths, results, stability_records, factor_records)
                            raise
                        finished += 1
                        results.extend(result["rows"])
                        stability_records.extend(result["stability"])
                        all_outcome_hashes.append(result["hash"])
                        factor_records.append(
                            _factor_record(result, task[0], task[1], scenario_order[task[0]])
                        )
                        progress["completed"] += 1
                        progress["case_completed"] = completed_before + finished
                        progress["last_task"] = {"scenario": task[0], "replicate": task[1]}
                        progress["last_task_elapsed_seconds"] = float(result.get("elapsed_seconds", np.nan))
                        progress["elapsed_seconds"] = float(time.perf_counter() - started_all)
                        should_checkpoint = (
                            checkpoint_paths is not None
                            and (finished % CHECKPOINT_EVERY == 0 or finished == len(tasks_pending))
                        )
                        if should_checkpoint:
                            _save_checkpoint(checkpoint_paths, results, stability_records, factor_records)
                            progress["last_checkpoint_utc"] = _utc_now()
                        if (
                            finished == len(tasks_pending)
                            or finished % max(1, min(20, len(tasks_pending))) == 0
                            or should_checkpoint
                        ):
                            _write_json(status_path, progress)
                            print(json.dumps({
                                "mode": mode,
                                "case": case,
                                "completed": completed_before + finished,
                                "planned": len(tasks),
                                "last_task": progress["last_task"],
                                "elapsed_seconds": progress["elapsed_seconds"],
                                "checkpoint_saved": bool(should_checkpoint),
                            }, ensure_ascii=False), flush=True)

            replications = pd.DataFrame(results)
            stability = pd.DataFrame(stability_records)
            summary, paired = _summaries(replications)
            method_paired = _inference_method_paired(replications)
            name = f"mvn_{mode}_{case}"
            rep_path = NEW_ROOT / "results" / f"{name}_replications.csv"
            factor_path = NEW_ROOT / "results" / f"{name}_factors.npz"
            stability_path = NEW_ROOT / "results" / f"{name}_stability.csv"
            repro_path = NEW_ROOT / "results" / f"{name}_reproducibility.json"
            summary_path = NEW_ROOT / "tables" / f"{name}_summary.csv"
            paired_path = NEW_ROOT / "tables" / f"{name}_paired.csv"
            method_paired_path = NEW_ROOT / "tables" / f"{name}_method_paired.csv"
            _write_csv(rep_path, replications)
            _write_csv(stability_path, stability)
            _write_csv(summary_path, summary)
            _write_csv(paired_path, paired)
            _write_csv(method_paired_path, method_paired)
            _save_factors(factor_path, factor_records)
            outputs = [
                str(path.relative_to(NEW_ROOT))
                for path in (
                    rep_path, factor_path, stability_path, summary_path, paired_path,
                    method_paired_path, repro_path,
                )
            ]
            repro = _case_repro(
                case,
                mode,
                case_started,
                reps_per_scenario,
                workers,
                replications,
                stability,
                case_state,
                outputs,
            )
            prior_case_seconds = 0.0
            if resume and prior_status.get("current_case") == case:
                prior_case_seconds = float(prior_status.get("elapsed_seconds") or 0.0)
            repro["previous_attempt_elapsed_seconds"] = prior_case_seconds
            repro["elapsed_seconds_including_previous_attempt"] = (
                float(repro["elapsed_seconds"]) + prior_case_seconds
            )
            if resume and prior_status.get("current_case") == case:
                repro["resume_history"] = {
                    "previous_pid": prior_status.get("pid"),
                    "previous_reported_completed": prior_status.get("case_completed"),
                    "resumed_checkpoint_completed": prior_status.get("checkpoint_completed_replications"),
                    "previous_uncheckpointed_delta_rerun": prior_status.get("reported_uncheckpointed_delta"),
                    "checkpoint_backup_sha256": prior_status.get("checkpoint_sha256"),
                    "checkpoint_backup_path": prior_status.get("backup_path"),
                }
            _write_json(repro_path, repro)
            expected_hashes = len(scenarios) * reps_per_scenario
            outcome_table = replications.drop_duplicates(["scenario", "replicate"])
            replay_rows = outcome_table
            covariance_validity = replications.drop_duplicates(
                ["covariance_method", "scenario", "replicate"]
            )
            valid_covariance_counts = {
                covariance_name: int(
                    covariance_validity.loc[
                        covariance_validity["covariance_method"].eq(covariance_name),
                        "covariance_psd_valid",
                    ].astype(bool).sum()
                )
                for covariance_name in ("quasi_pearson", "hac28")
            }
            case_check = {
                "case": case,
                "mode": mode,
                "planned_outcomes": expected_hashes,
                "observed_outcomes": int(len(outcome_table)),
                "unique_outcome_hashes": int(outcome_table["outcome_sha256"].nunique()),
                "all_outcome_hashes_unique": bool(outcome_table["outcome_sha256"].nunique() == expected_hashes),
                "all_replays_converged": bool(replay_rows["converged"].astype(bool).all()),
                "all_A_hat_matches_1e_8": bool(replay_rows["match_A_hat_1e_8"].astype(bool).all()),
                "all_model_SE_matches_1e_8": bool(replay_rows["match_se_model_1e_8"].astype(bool).all()),
                "max_relative_A_hat_difference": float(pd.to_numeric(replay_rows["relative_A_hat_difference"], errors="coerce").max()),
                "max_relative_model_SE_difference": float(pd.to_numeric(replay_rows["relative_se_model_difference"], errors="coerce").max()),
                "n_valid_covariances": valid_covariance_counts,
                "n_nonpsd_hac28_covariances": int(
                    (~covariance_validity.loc[
                        covariance_validity["covariance_method"].eq("hac28"),
                        "covariance_psd_valid",
                    ].astype(bool)).sum()
                ),
                "elapsed_seconds": float(time.perf_counter() - case_started),
                "outputs": outputs,
            }
            case_checks.append(case_check)
            qa_path = NEW_ROOT / "qa" / f"mvn_{mode}_checks.json"
            checks = {
                "mode": mode,
                "updated_utc": _utc_now(),
                "all_expected_outcome_hashes_unique_across_cases_so_far": len(set(all_outcome_hashes)) == len(all_outcome_hashes),
                "cases": case_checks,
            }
            _write_json(qa_path, checks)
            if checkpoint_paths is not None:
                for checkpoint_path in checkpoint_paths.values():
                    resolved = checkpoint_path.resolve()
                    if NEW_ROOT.resolve() in resolved.parents and checkpoint_path.is_file():
                        checkpoint_path.unlink()

        final_checks = {
            "mode": mode,
            "completed_utc": _utc_now(),
            "all_outcome_hashes_unique_across_cases": len(set(all_outcome_hashes)) == len(all_outcome_hashes),
            "n_outcome_hashes": len(all_outcome_hashes),
            "cases": case_checks,
        }
        _write_json(NEW_ROOT / "qa" / f"mvn_{mode}_checks.json", final_checks)
        progress.update({
            "status": "completed",
            "completed_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started_all),
            "checks_path": str(NEW_ROOT / "qa" / f"mvn_{mode}_checks.json"),
        })
        _write_json(status_path, progress)
        return final_checks
    except Exception as exc:
        progress.update({
            "status": "failed",
            "failed_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started_all),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        _write_json(status_path, progress)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--case", choices=("state", "city", "both"), default="both")
    parser.add_argument("--pilot-reps", type=int, default=1, help="pilot replicates per scenario (1–4)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--formal-approved", action="store_true", help="required before a formal 6000-outcome run")
    parser.add_argument("--resume", action="store_true", help="resume missing formal replications from checkpoint")
    parser.add_argument("--resummarize-existing", action="store_true", help="refresh tables without replaying outcomes")
    args = parser.parse_args()
    if args.mode == "formal" and not args.formal_approved:
        parser.error("formal execution requires explicit root approval via --formal-approved")
    try:
        if args.resummarize_existing:
            if args.resume:
                parser.error("--resume cannot be combined with --resummarize-existing")
            result = resummarize_existing(args.mode, args.case)
        else:
            result = run(args.mode, args.case, args.pilot_reps, args.workers, resume=args.resume)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
