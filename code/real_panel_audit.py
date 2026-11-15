#!/usr/bin/env python3
"""Daily real-panel audit for the fixed 1 C temperature-path contrast.

The estimand is the model-based expected death-count difference between the
observed São Paulo daily temperature path and that same path shifted down by
1 C, over a shared outcome window. This is a conditional attribution contrast,
not a causal or climate-attribution claim.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any

# Keep numerical libraries single-threaded; the process pool is capped at four.
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline
from scipy.linalg import LinAlgError, lstsq, solve, svdvals
from scipy.special import gammaln
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "inputs" / "public" / "bra_state_day.parquet"
CONFIG_PATH = PROJECT_ROOT / "inputs" / "real_config.json"
RESULTS_DIR = output_path(PROJECT_ROOT / "results")
DOCS_DIR = output_path(PROJECT_ROOT / "docs")
LEGACY_PANEL_SHA256 = "fc972f919a5e5d51280a4093b4992d2b5ea7add6c51c2193b741d412825a42d3"

_WORKER_STATE: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def read_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Real-audit config is required: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("study_id") != "real_daily_panel_audit_v1":
        raise ValueError("Unexpected real-audit study_id.")
    return config


def _natural_cardinal_values(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic cardinal basis with linear continuation beyond boundaries."""
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1 or len(knots) < 2 or not np.all(np.diff(knots) > 0):
        raise ValueError("Natural-spline knots must be a strictly increasing vector.")
    spline = CubicSpline(knots, np.eye(len(knots)), axis=0, bc_type="natural", extrapolate=False)
    clipped = np.clip(x, knots[0], knots[-1])
    values = np.asarray(spline(clipped), dtype=float)
    low = x < knots[0]
    high = x > knots[-1]
    if np.any(low):
        values[low] += (x[low] - knots[0])[:, None] * spline(knots[0], 1)
    if np.any(high):
        values[high] += (x[high] - knots[-1])[:, None] * spline(knots[-1], 1)
    return values


@dataclass(frozen=True)
class CenteredSplineSpec:
    knots: tuple[float, ...]
    center: tuple[float, ...]
    rotation: tuple[tuple[float, ...], ...]
    columns: int

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raw = _natural_cardinal_values(np.asarray(x, dtype=float), np.asarray(self.knots, dtype=float))
        centered = raw - np.asarray(self.center, dtype=float)
        return centered @ np.asarray(self.rotation, dtype=float)

    def as_dict(self) -> dict[str, Any]:
        return {
            "knots": list(self.knots),
            "center": list(self.center),
            "rotation": [list(row) for row in self.rotation],
            "columns": self.columns,
        }


def _fit_centered_cardinal_spec(x: np.ndarray, df: int, knot_values: np.ndarray | None = None) -> CenteredSplineSpec:
    """Fit a centered rank-df basis from df+1 cardinal knots."""
    x = np.asarray(x, dtype=float)
    if df < 1:
        raise ValueError("Spline degrees of freedom must be positive.")
    if knot_values is None:
        knot_values = np.linspace(float(x.min()), float(x.max()), df + 1)
    knots = np.asarray(knot_values, dtype=float)
    if len(knots) != df + 1:
        raise ValueError("Centered cardinal construction requires exactly df+1 knots.")
    raw = _natural_cardinal_values(x, knots)
    center = raw.mean(axis=0)
    centered = raw - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tol = max(centered.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 1.0) * 100
    rank = int(np.sum(singular > tol))
    if rank != df:
        raise ValueError(f"Centered spline basis rank {rank} differs from requested df {df}.")
    rotation = vt[:df].T
    return CenteredSplineSpec(
        tuple(float(v) for v in knots),
        tuple(float(v) for v in center),
        tuple(tuple(float(v) for v in row) for row in rotation),
        df,
    )


def _fit_temperature_spec(temperature: np.ndarray) -> CenteredSplineSpec:
    """Two-column natural spline using factual 0th/50th/100th percentile knots."""
    temperature = np.asarray(temperature, dtype=float)
    if not np.all(np.isfinite(temperature)):
        raise ValueError("Temperature values must be finite.")
    knots = np.quantile(temperature, [0.0, 0.5, 1.0])
    if np.any(np.diff(knots) <= 1e-8):
        raise ValueError("Temperature has insufficient variation for the two-column spline.")
    return _fit_centered_cardinal_spec(temperature, df=2, knot_values=knots)


def _lag_basis(max_lag: int) -> np.ndarray:
    """Three-column natural spline basis over integer lag 0..max_lag."""
    if max_lag < 2:
        raise ValueError("max_lag must be at least 2.")
    knots = np.linspace(0.0, float(max_lag), 3)
    return _natural_cardinal_values(np.arange(max_lag + 1, dtype=float), knots)


def _crossbasis(temp: np.ndarray, temp_spec: CenteredSplineSpec, max_lag: int) -> np.ndarray:
    """Return (outcome day, lag-basis component, temperature basis) columns."""
    temp = np.asarray(temp, dtype=float)
    if temp.ndim != 1 or len(temp) <= max_lag:
        raise ValueError("Temperature series is too short for requested lag horizon.")
    basis = temp_spec.evaluate(temp)
    lag = _lag_basis(max_lag)
    rows = np.arange(max_lag, len(temp))
    out = np.zeros((len(rows), lag.shape[1], basis.shape[1]), dtype=float)
    for lag_value in range(max_lag + 1):
        exposure_at_lag = basis[rows - lag_value]
        out += lag[lag_value][None, :, None] * exposure_at_lag[:, None, :]
    return out.reshape(len(rows), -1, order="C")


def _standardize_design(
    factual: np.ndarray, counterfactual: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    factual = np.asarray(factual, dtype=float)
    counterfactual = np.asarray(counterfactual, dtype=float)
    if factual.shape != counterfactual.shape or factual.shape[1] != len(names):
        raise ValueError("Factual/counterfactual design dimensions do not match column names.")
    if not np.all(np.isfinite(factual)) or not np.all(np.isfinite(counterfactual)):
        raise ValueError("Design contains non-finite entries.")
    center = factual[:, 1:].mean(axis=0)
    centered = factual[:, 1:] - center
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        bad = [names[i + 1] for i, value in enumerate(scale) if not np.isfinite(value) or value <= 1e-12]
        raise ValueError(f"Design contains constant or invalid non-intercept columns: {bad}")
    X = np.column_stack((np.ones(len(factual)), centered / scale))
    Xcf = np.column_stack((np.ones(len(counterfactual)), (counterfactual[:, 1:] - center) / scale))
    sv = svdvals(X, check_finite=False)
    rank_tol = max(X.shape) * np.finfo(float).eps * float(sv[0]) * 100.0
    rank = int(np.sum(sv > rank_tol))
    if rank != X.shape[1]:
        raise ValueError(f"Standardized design is not full rank: rank {rank} of {X.shape[1]}.")
    meta = {
        "column_names": names,
        "center": [float(v) for v in center],
        "scale_rms": [float(v) for v in scale],
        "rank": rank,
        "columns": int(X.shape[1]),
        "singular_value_condition_number": float(sv[0] / sv[-1]),
        "rank_tolerance": float(rank_tol),
    }
    return X, Xcf, meta


def build_design(
    panel: pd.DataFrame,
    *,
    max_lag: int,
    time_df: int,
    trim_days: int,
    cf_shift_c: float,
) -> dict[str, Any]:
    """Construct a centered full-rank daily cross-basis design.

    Column order is intercept; six cross-basis columns grouped as two
    temperature basis columns within each of three temporal lag-basis
    components; time natural-spline columns; and six centered weekday columns.
    The temporal lag-basis groups are basis components, not physical lag days.
    """
    if trim_days < max_lag:
        raise ValueError("trim_days must be at least max_lag for a common complete-history window.")
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(panel["date"])
    deaths = panel["deaths"].to_numpy(dtype=float)
    temperature = panel["tmean"].to_numpy(dtype=float)
    if len(panel) <= trim_days or np.any(~np.isfinite(deaths)) or np.any(deaths < 0):
        raise ValueError("Outcome panel is invalid or too short.")
    if not np.allclose(deaths, np.round(deaths)) or not np.all(np.diff(dates.to_numpy()) == np.timedelta64(1, "D")):
        raise ValueError("Daily outcomes must be integer counts on a complete unique calendar.")
    if dates.duplicated().any():
        raise ValueError("Duplicate dates in the daily panel.")

    temp_spec = _fit_temperature_spec(temperature)
    cf_temperature = temperature - float(cf_shift_c)
    cb_all = _crossbasis(temperature, temp_spec, max_lag)
    cb_cf_all = _crossbasis(cf_temperature, temp_spec, max_lag)
    # _crossbasis starts at max_lag; align to the single window trimmed at 21 days.
    all_outcome_indices = np.arange(max_lag, len(panel))
    keep = all_outcome_indices >= trim_days
    row_indices = all_outcome_indices[keep]
    cb = cb_all[keep]
    cb_cf = cb_cf_all[keep]

    # Natural cubic time smooth: exactly time_df centered columns, seven/year at
    # the primary 35 df over the prespecified five calendar years.
    date_origin = dates.iloc[0]
    time_days = (dates.iloc[row_indices] - date_origin).dt.days.to_numpy(dtype=float)
    time_spec = _fit_centered_cardinal_spec(time_days, df=time_df)
    time_basis = time_spec.evaluate(time_days)
    weekdays = dates.iloc[row_indices].dt.dayofweek.to_numpy(dtype=int)
    dow = np.column_stack([(weekdays == d).astype(float) for d in range(1, 7)])

    names = ["intercept"]
    names.extend(
        f"cb_lag_basis_{j + 1}_temp_basis_{k + 1}"
        for j in range(3) for k in range(2)
    )
    names.extend(f"time_ns_{j + 1}" for j in range(time_df))
    names.extend(f"dow_{d}" for d in range(1, 7))
    factual_raw = np.column_stack((np.ones(len(row_indices)), cb, time_basis, dow))
    cf_raw = np.column_stack((np.ones(len(row_indices)), cb_cf, time_basis, dow))
    X, Xcf, design_meta = _standardize_design(factual_raw, cf_raw, names)
    y = deaths[row_indices].astype(np.int64)
    off = np.zeros(len(y), dtype=float)
    cb_start = 1
    lag_blocks = [
        {"name": f"lag_basis_component_{j + 1}", "columns": [cb_start + 2 * j, cb_start + 2 * j + 1]}
        for j in range(3)
    ]
    mapping = {
        "design_source": "real_panel_audit.build_design daily temperature × natural-spline temporal lag cross-basis",
        "basis_names": ["temperature_ns_1", "temperature_ns_2"],
        "lag_blocks": lag_blocks,
        "block_semantics": (
            "Each mapped two-column group is one coefficient pair for a temporal "
            "lag-basis component. These groups are not physical lag days. A future "
            "bounded-sensitivity analysis must declare and justify allocation "
            "weights across these basis components."
        ),
        "flattening_order": "lag_basis_component_major_then_temperature_basis",
    }
    return {
        "X": X,
        "Xcf": Xcf,
        "y": y,
        "offset": off,
        "dates": dates.iloc[row_indices].dt.strftime("%Y-%m-%d").to_numpy(),
        "row_indices": row_indices,
        "design_meta": design_meta,
        "column_names": names,
        "temperature_spec": temp_spec,
        "time_spec": time_spec,
        "lag_basis": _lag_basis(max_lag),
        "max_lag": int(max_lag),
        "time_df": int(time_df),
        "trim_days": int(trim_days),
        "cf_shift_c": float(cf_shift_c),
        "block_metadata": mapping,
        "estimand_window": [str(dates.iloc[row_indices[0]].date()), str(dates.iloc[row_indices[-1]].date())],
        "n_rows": int(len(y)),
    }


def _poisson_loglik(y: np.ndarray, eta: np.ndarray) -> float:
    if not np.all(np.isfinite(eta)) or np.max(eta) >= 700.0:
        return -math.inf
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mu = np.exp(eta)
    if not np.all(np.isfinite(mu)):
        return -math.inf
    return float(np.sum(y * eta - mu - gammaln(y + 1.0)))


def _bread_and_meat(X: np.ndarray, y: np.ndarray, mu: np.ndarray, bandwidth: int) -> tuple[np.ndarray, np.ndarray]:
    n, p = X.shape
    weighted = X * np.sqrt(mu)[:, None]
    hessian = weighted.T @ weighted
    bread = solve(hessian, np.eye(p), assume_a="pos", check_finite=False)
    score = X * (y - mu)[:, None]
    meat = score.T @ score
    max_lag = min(int(bandwidth), n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    meat = (meat + meat.T) / 2.0
    return bread, meat


def fit_poisson_irls(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray | None = None,
    *,
    Xcf: np.ndarray | None = None,
    hac_bandwidths: tuple[int, ...] = (14, 28),
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Fit Poisson log-mean GLM by line-searched Fisher scoring / IRLS.

    Returns beta, fitted means, the model-based covariance, the sandwich
    covariance for each requested Bartlett HAC score bandwidth, and diagnostics.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.zeros(len(y), dtype=float) if offset is None else np.asarray(offset, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y) or offset.shape != y.shape:
        raise ValueError("X, y, and offset dimensions do not align.")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)) or np.any(y < 0) or np.any(~np.isfinite(offset)):
        raise ValueError("IRLS inputs must be finite and outcomes nonnegative.")
    if Xcf is not None:
        Xcf = np.asarray(Xcf, dtype=float)
        if Xcf.shape != X.shape or np.any(~np.isfinite(Xcf)):
            raise ValueError("Xcf must be finite and have the same dimensions as X.")
    singular = svdvals(X, check_finite=False)
    rank_tol = max(X.shape) * np.finfo(float).eps * float(singular[0]) * 100.0
    rank = int(np.sum(singular > rank_tol))
    if rank != X.shape[1]:
        raise ValueError(f"Poisson design is rank deficient: rank {rank} of {X.shape[1]}.")

    beta = np.zeros(X.shape[1], dtype=float)
    if X.shape[1] and np.allclose(X[:, 0], 1.0):
        beta[0] = math.log(max(float(np.mean(y)), 1e-6))
    eta = offset + X @ beta
    loglik = _poisson_loglik(y, eta)
    converged = False
    reason = "iteration_limit"
    accepted_iterations = 0
    last_step = math.inf
    for iteration in range(1, max_iterations + 1):
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            mu = np.exp(eta)
        if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
            reason = "nonfinite_mean"
            break
        sqrt_w = np.sqrt(mu)
        working = (eta - offset) + (y - mu) / mu
        candidate, _, candidate_rank, _ = lstsq(
            X * sqrt_w[:, None], working * sqrt_w, cond=1e-12,
            lapack_driver="gelsy", check_finite=False,
        )
        if int(candidate_rank) != X.shape[1] or np.any(~np.isfinite(candidate)):
            reason = "weighted_design_rank_failure"
            break
        delta = candidate - beta
        accepted = False
        step = 1.0
        for _ in range(22):
            trial = beta + step * delta
            trial_eta = offset + X @ trial
            trial_ll = _poisson_loglik(y, trial_eta)
            if trial_ll >= loglik - 1e-10 * (1.0 + abs(loglik)):
                beta = trial
                eta = trial_eta
                last_step = float(np.max(np.abs(step * delta)))
                loglik = trial_ll
                accepted_iterations = iteration
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if float(np.max(np.abs(delta))) <= tolerance * 10:
                converged = True
                reason = "score_stationary"
            else:
                reason = "line_search_failed"
            break
        if last_step <= tolerance:
            converged = True
            reason = "step_tolerance"
            break
    if not converged and accepted_iterations == max_iterations:
        reason = "iteration_limit"

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mu = np.exp(eta)
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise FloatingPointError("Final Poisson means are non-finite or nonpositive.")
    weighted = X * np.sqrt(mu)[:, None]
    hessian = weighted.T @ weighted
    try:
        bread = solve(hessian, np.eye(X.shape[1]), assume_a="pos", check_finite=False)
    except LinAlgError as exc:
        raise np.linalg.LinAlgError("Final Poisson information matrix is singular.") from exc
    bread = (bread + bread.T) / 2.0
    condition = float(np.linalg.cond(hessian))
    beta_covariance = bread.copy()
    se: dict[str, float | None] = {}
    covariance_a: dict[str, float | None] = {}
    a_hat: float | None = None
    model_var_a: float | None = None
    gradient_a = None
    mu_cf = None
    if Xcf is not None:
        eta_cf = offset + Xcf @ beta
        if not np.all(np.isfinite(eta_cf)) or np.max(eta_cf) >= 700:
            raise FloatingPointError("Counterfactual linear predictor is outside finite exponential range.")
        mu_cf = np.exp(eta_cf)
        a_hat = float(np.sum(mu - mu_cf))
        gradient_a = X.T @ mu - Xcf.T @ mu_cf
        model_var_a = float(gradient_a @ bread @ gradient_a)
        if model_var_a < -1e-9:
            raise FloatingPointError(f"Model-based attribution variance is negative ({model_var_a}).")
        model_var_a = max(0.0, model_var_a)
        se["model"] = math.sqrt(model_var_a)
        covariance_a["model"] = model_var_a
    for bandwidth in hac_bandwidths:
        _, meat = _bread_and_meat(X, y, mu, int(bandwidth))
        cov = bread @ meat @ bread
        cov = (cov + cov.T) / 2.0
        beta_covariance = cov
        key = f"hac{int(bandwidth)}"
        if gradient_a is not None:
            variance = float(gradient_a @ cov @ gradient_a)
            if variance < -1e-8:
                raise FloatingPointError(f"{key} attribution variance is negative ({variance}).")
            variance = max(0.0, variance)
            covariance_a[key] = variance
            se[key] = math.sqrt(variance)
    score = X.T @ (y - mu)
    return {
        "beta": beta,
        "mu": mu,
        "mu_cf": mu_cf,
        "A_hat": a_hat,
        "se": se,
        "variance_A": covariance_a,
        "beta_covariance_last_hac": beta_covariance,
        "loglik": float(loglik),
        "converged": bool(converged),
        "convergence_reason": reason,
        "iterations": int(accepted_iterations),
        "last_max_abs_step": float(last_step),
        "max_abs_score": float(np.max(np.abs(score))),
        "weighted_information_condition_number": condition,
        "design_rank": rank,
        "n": int(len(y)),
        "p": int(X.shape[1]),
    }


def _model_interval(a_hat: float, se: float, alpha: float) -> tuple[float, float]:
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    half = critical * se
    return float(a_hat - half), float(a_hat + half)


def _interaction_component(dates: np.ndarray, temperature: np.ndarray, amplitude: float) -> np.ndarray:
    parsed = pd.to_datetime(dates)
    doy = parsed.dayofyear.to_numpy(dtype=float)
    seasonal = np.sin(2.0 * np.pi * (doy - 1.0) / 365.25)
    return float(amplitude) * ((np.asarray(temperature, dtype=float) - 20.0) / 10.0) * seasonal


def _lognormal_variance_from_residuals(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    residual_sum_squares = float(np.sum((y - mu) ** 2))
    poisson_variance = float(np.sum(mu))
    mean_square_sum = float(np.sum(mu ** 2))
    if mean_square_sum <= 0:
        return 0.0
    return max(math.log1p(max(residual_sum_squares - poisson_variance, 0.0) / mean_square_sum), 0.0)


def _init_worker(state: dict[str, Any]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _one_simulation_rep(task: tuple[str, int]) -> dict[str, Any]:
    scenario, replicate = task
    if _WORKER_STATE is None:
        raise RuntimeError("Simulation worker state was not initialized.")
    state = _WORKER_STATE
    seed_parts = [
        int(state["base_seed"]), int(state["mode_id"]),
        int(state["scenario_ids"][scenario]), int(replicate),
    ]
    seed_seq = np.random.SeedSequence(seed_parts)
    rng = np.random.default_rng(seed_seq)
    means = state["means"][scenario]
    if scenario == "correlated_overdispersed_ar1":
        sigma2 = float(state["lognormal_sigma2"])
        phi = float(state["latent_ar1_phi"])
        if sigma2 <= 0:
            latent = np.zeros(len(means), dtype=float)
        else:
            latent = np.empty(len(means), dtype=float)
            latent[0] = rng.normal(0.0, math.sqrt(sigma2))
            innovation_sd = math.sqrt(sigma2 * (1.0 - phi * phi))
            for index in range(1, len(means)):
                latent[index] = phi * latent[index - 1] + rng.normal(0.0, innovation_sd)
        conditional_mean = means * np.exp(latent - 0.5 * sigma2)
        y = rng.poisson(conditional_mean).astype(float)
    else:
        y = rng.poisson(means).astype(float)
    try:
        fit = fit_poisson_irls(
            state["X"], y, state["offset"], Xcf=state["Xcf"],
            hac_bandwidths=(14, 28), max_iterations=int(state["max_iterations"]),
            tolerance=float(state["tolerance"]),
        )
        if not fit["converged"]:
            return {
                "scenario": scenario, "replicate": replicate, "seed_entropy": seed_parts,
                "converged": False, "failure_reason": fit["convergence_reason"],
                "iterations": fit["iterations"], "A_hat": None, "A_true": state["truth"][scenario],
                "se_model": None, "se_hac14": None, "se_hac28": None,
                "cover_model": False, "cover_hac14": False, "cover_hac28": False,
                "width_model": None, "width_hac14": None, "width_hac28": None,
            }
        record: dict[str, Any] = {
            "scenario": scenario,
            "replicate": int(replicate),
            "seed_entropy": seed_parts,
            "converged": True,
            "failure_reason": "",
            "iterations": fit["iterations"],
            "A_hat": float(fit["A_hat"]),
            "A_true": float(state["truth"][scenario]),
        }
        for method, key in (("model", "model"), ("hac14", "hac14"), ("hac28", "hac28")):
            se = float(fit["se"][key])
            lo, hi = _model_interval(float(fit["A_hat"]), se, float(state["alpha"]))
            record[f"se_{method}"] = se
            record[f"cover_{method}"] = bool(lo <= state["truth"][scenario] <= hi)
            record[f"width_{method}"] = float(hi - lo)
        return record
    except Exception as exc:  # Preserve failed replicates as planned uncovered outcomes.
        return {
            "scenario": scenario, "replicate": int(replicate),
            "seed_entropy": seed_parts,
            "converged": False, "failure_reason": f"{type(exc).__name__}: {exc}",
            "iterations": None, "A_hat": None, "A_true": float(state["truth"][scenario]),
            "se_model": None, "se_hac14": None, "se_hac28": None,
            "cover_model": False, "cover_hac14": False, "cover_hac28": False,
            "width_model": None, "width_hac14": None, "width_hac28": None,
        }


def _run_simulations(
    config: dict[str, Any], design: dict[str, Any], fit_obs: dict[str, Any],
    panel: pd.DataFrame, mode: str, workers: int, replications: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = config["simulation"]
    reps = int(sim[f"{mode}_reps_per_scenario"] if replications is None else replications)
    if reps < 1:
        raise ValueError("Simulation repetitions must be positive.")
    alpha = float(config["inference"]["alpha"])
    means_compatible = np.asarray(fit_obs["mu"], dtype=float)
    cf_mean_compatible = np.asarray(fit_obs["mu_cf"], dtype=float)
    if np.any(means_compatible <= 0) or np.any(cf_mean_compatible <= 0):
        raise ValueError("Observed fitted means are invalid for fixed-design simulation.")

    row_indices = np.asarray(design["row_indices"], dtype=int)
    all_dates = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    all_temp = panel["tmean"].to_numpy(dtype=float)
    selected_dates = all_dates[row_indices]
    selected_temp = all_temp[row_indices]
    cf_temp = selected_temp - float(design["cf_shift_c"])
    amplitude = float(sim["omitted_interaction"]["amplitude"])
    h = _interaction_component(selected_dates, selected_temp, amplitude)
    h_cf = _interaction_component(selected_dates, cf_temp, amplitude)
    means_omitted = means_compatible * np.exp(h)
    cf_means_omitted = cf_mean_compatible * np.exp(h_cf)
    truths = {
        "model_compatible": float(np.sum(means_compatible - cf_mean_compatible)),
        "smooth_omitted_interaction": float(np.sum(means_omitted - cf_means_omitted)),
        "correlated_overdispersed_ar1": float(np.sum(means_compatible - cf_mean_compatible)),
    }
    means = {
        "model_compatible": means_compatible,
        "smooth_omitted_interaction": means_omitted,
        "correlated_overdispersed_ar1": means_compatible,
    }
    scenarios = sim["scenarios"]
    name_to_id = {item["name"]: int(item["seed_id"]) for item in scenarios}
    if set(means) != set(name_to_id):
        raise ValueError("Simulation scenario names differ from the declared configuration.")
    lognormal_sigma2 = _lognormal_variance_from_residuals(design["y"], means_compatible)
    state = {
        "X": design["X"], "Xcf": design["Xcf"], "offset": design["offset"],
        "means": means, "truth": truths, "alpha": alpha,
        "base_seed": int(sim["base_seed"]), "mode_id": int(sim["mode_ids"][mode]),
        "scenario_ids": name_to_id,
        "lognormal_sigma2": lognormal_sigma2,
        "latent_ar1_phi": float(sim["correlated_overdispersed"]["phi"]),
        "max_iterations": int(config["fit"]["max_iterations"]),
        "tolerance": float(config["fit"]["tolerance_max_abs_step"]),
    }
    tasks = [(name, rep) for name in means for rep in range(reps)]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, int(config["execution"]["max_workers"])),
            initializer=_init_worker, initargs=(state,),
        ) as executor:
            rows = list(executor.map(_one_simulation_rep, tasks, chunksize=1))
    else:
        _init_worker(state)
        rows = [_one_simulation_rep(task) for task in tasks]
    results = pd.DataFrame(rows)
    summaries = []
    for scenario in means:
        group = results.loc[results["scenario"].eq(scenario)]
        success = group.loc[group["converged"]]
        for method in ("model", "hac14", "hac28"):
            cover_col = f"cover_{method}"
            covered = int(group[cover_col].sum())
            n_plan = len(group)
            n_ok = len(success)
            success_cov = int(success[cover_col].sum())
            width = pd.to_numeric(success[f"width_{method}"], errors="coerce").dropna()
            summaries.append({
                "scenario": scenario,
                "method": method,
                "A_true": truths[scenario],
                "n_planned": int(n_plan),
                "n_converged": int(n_ok),
                "n_failed": int(n_plan - n_ok),
                "covered_planned": covered,
                "coverage_planned_failures_uncovered": float(covered / n_plan),
                "coverage_mcse_planned": float(math.sqrt((covered / n_plan) * (1.0 - covered / n_plan) / n_plan)),
                "covered_converged": success_cov,
                "coverage_converged_only": float(success_cov / n_ok) if n_ok else None,
                "coverage_mcse_converged_only": (
                    float(math.sqrt((success_cov / n_ok) * (1.0 - success_cov / n_ok) / n_ok)) if n_ok else None
                ),
                "mean_interval_width_converged": float(width.mean()) if len(width) else None,
                "mean_bias_converged": (
                    float((success["A_hat"] - success["A_true"]).mean()) if n_ok else None
                ),
            })
    paired_rows = []
    method_pairs = (("model", "hac14"), ("model", "hac28"), ("hac14", "hac28"))
    for scenario in means:
        group = results.loc[results["scenario"].eq(scenario)].copy()
        for first, second in method_pairs:
            diff_cover = group[f"cover_{first}"].astype(float) - group[f"cover_{second}"].astype(float)
            pair_ok = group["converged"].astype(bool)
            diff_width = (
                pd.to_numeric(group.loc[pair_ok, f"width_{first}"], errors="coerce")
                - pd.to_numeric(group.loc[pair_ok, f"width_{second}"], errors="coerce")
            ).dropna()
            paired_rows.append({
                "scenario": scenario,
                "method_1": first,
                "method_2": second,
                "n_planned": int(len(group)),
                "mean_paired_coverage_difference": float(diff_cover.mean()),
                "mcse_paired_coverage_difference": float(diff_cover.std(ddof=1) / math.sqrt(len(diff_cover))) if len(diff_cover) > 1 else None,
                "n_converged_width_pairs": int(len(diff_width)),
                "mean_paired_width_difference": float(diff_width.mean()) if len(diff_width) else None,
                "mcse_paired_width_difference": float(diff_width.std(ddof=1) / math.sqrt(len(diff_width))) if len(diff_width) > 1 else None,
            })
    return results, pd.DataFrame(summaries), pd.DataFrame(paired_rows)


def _write_design_artifact(path: Path, design: dict[str, Any], fit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mapping_json = json.dumps(design["block_metadata"], ensure_ascii=False)
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=design["X"],
            Xcf=design["Xcf"],
            y=design["y"],
            offset=design["offset"],
            beta=fit["beta"],
            outcome_dates=design["dates"],
            column_names=np.asarray(design["column_names"], dtype=str),
            block_metadata_json=np.asarray(mapping_json),
            design_meta_json=np.asarray(json.dumps(design["design_meta"], ensure_ascii=False)),
            temperature_spec_json=np.asarray(json.dumps(design["temperature_spec"].as_dict(), ensure_ascii=False)),
            time_spec_json=np.asarray(json.dumps(design["time_spec"].as_dict(), ensure_ascii=False)),
            lag_basis=design["lag_basis"],
        )
    tmp.replace(path)


def _sensitivity_table(config: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fit_config = config["fit"]
    infer = config["inference"]
    base_lag = int(config["design"]["primary_max_lag"])
    base_time_df = int(config["design"]["primary_time_df_per_year"])
    trim = int(config["design"]["common_trim_days"])
    cf_shift = float(config["estimand"]["counterfactual_shift_c"])
    for time_df in sorted(set([base_time_df, *config["sensitivity"]["time_df_per_year"]])):
        # five-year calendar span is fixed in config; exact requested df follows 5 * df/year.
        total_df = int(time_df * int(config["design"]["study_years"]))
        for max_lag in sorted(set([base_lag, *config["sensitivity"]["max_lag_days"]])):
            design = build_design(
                panel, max_lag=int(max_lag), time_df=total_df,
                trim_days=trim, cf_shift_c=cf_shift,
            )
            fit = fit_poisson_irls(
                design["X"], design["y"], design["offset"], Xcf=design["Xcf"],
                hac_bandwidths=tuple(int(x) for x in infer["hac_bandwidths"]),
                max_iterations=int(fit_config["max_iterations"]),
                tolerance=float(fit_config["tolerance_max_abs_step"]),
            )
            for method, se_key in (("model", "model"), ("hac14", "hac14"), ("hac28", "hac28")):
                se = float(fit["se"][se_key])
                lo, hi = _model_interval(float(fit["A_hat"]), se, float(infer["alpha"]))
                rows.append({
                    "time_df_per_year": int(time_df),
                    "time_df_total": total_df,
                    "max_lag_days": int(max_lag),
                    "common_trim_days": trim,
                    "lag_basis_dimension": 3,
                    "method": method,
                    "n_outcomes": fit["n"],
                    "converged": fit["converged"],
                    "convergence_reason": fit["convergence_reason"],
                    "iterations": fit["iterations"],
                    "weighted_information_condition_number": fit["weighted_information_condition_number"],
                    "A_hat": float(fit["A_hat"]),
                    "standard_error": se,
                    "interval_level": float(1.0 - infer["alpha"]),
                    "interval_lo": lo,
                    "interval_hi": hi,
                })
    return pd.DataFrame(rows)


def run(mode: str = "pilot", workers: int = 4, pilot_reps: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in {"pilot", "formal"}:
        raise ValueError("mode must be 'pilot' or 'formal'.")
    if workers < 1:
        raise ValueError("workers must be at least one.")
    if pilot_reps is not None and (mode != "pilot" or pilot_reps < 1):
        raise ValueError("pilot_reps is only valid as a positive pilot-mode override.")
    config = read_config()
    if mode == "formal" and config["protocol_status"] != "approved":
        raise PermissionError("Formal simulation is blocked until the root approves real_config.json.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Public aggregate panel missing: {DATA_PATH}")
    panel_sha = sha256_file(DATA_PATH)
    if panel_sha.lower() != LEGACY_PANEL_SHA256:
        raise ValueError("Aggregate-panel SHA256 differs from public_data.py source manifest.")
    data = pd.read_parquet(DATA_PATH)
    required = {"state", "date", "deaths", "tmean"}
    if set(data.columns) != required:
        raise ValueError(f"Unexpected aggregate schema {list(data.columns)}")
    data["state"] = data["state"].astype(str).str.zfill(2)
    data["date"] = pd.to_datetime(data["date"])
    design_cfg = config["design"]
    panel = data.loc[
        data["state"].eq(str(design_cfg["state_code"]).zfill(2))
        & data["date"].dt.year.between(int(design_cfg["start_year"]), int(design_cfg["end_year"]))
    ].sort_values("date").reset_index(drop=True)
    expected_days = int((panel["date"].max() - panel["date"].min()).days + 1)
    if len(panel) != expected_days or panel["date"].duplicated().any():
        raise ValueError(f"Selected state panel is not complete: {len(panel)} rows for {expected_days} dates.")
    if int(panel["date"].dt.year.min()) != int(design_cfg["start_year"]) or int(panel["date"].dt.year.max()) != int(design_cfg["end_year"]):
        raise ValueError("Selected panel date range differs from config.")

    base = build_design(
        panel,
        max_lag=int(design_cfg["primary_max_lag"]),
        time_df=int(design_cfg["primary_time_df_per_year"]) * int(design_cfg["study_years"]),
        trim_days=int(design_cfg["common_trim_days"]),
        cf_shift_c=float(config["estimand"]["counterfactual_shift_c"]),
    )
    fit_cfg = config["fit"]
    infer = config["inference"]
    observed = fit_poisson_irls(
        base["X"], base["y"], base["offset"], Xcf=base["Xcf"],
        hac_bandwidths=tuple(int(x) for x in infer["hac_bandwidths"]),
        max_iterations=int(fit_cfg["max_iterations"]),
        tolerance=float(fit_cfg["tolerance_max_abs_step"]),
    )
    if not observed["converged"]:
        raise RuntimeError(f"Observed primary fit did not converge: {observed['convergence_reason']}")
    lognormal_sigma2 = _lognormal_variance_from_residuals(base["y"], observed["mu"])
    alpha = float(infer["alpha"])
    interval_rows = []
    for method in ("model", "hac14", "hac28"):
        se = float(observed["se"][method])
        lo, hi = _model_interval(float(observed["A_hat"]), se, alpha)
        interval_rows.append({
            "method": method,
            "A_hat_deaths": float(observed["A_hat"]),
            "standard_error": se,
            "interval_level": 1.0 - alpha,
            "normal_critical_value": float(norm.ppf(1.0 - alpha / 2.0)),
            "interval_lo": lo,
            "interval_hi": hi,
            "n_outcomes": observed["n"],
        })
    primary_report = {
        "analysis": "model-conditional daily attribution contrast",
        "state_code": str(design_cfg["state_code"]).zfill(2),
        "period": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "outcome_window": base["estimand_window"],
        "n_calendar_days": int(len(panel)),
        "n_outcome_days": int(base["n_rows"]),
        "cf_path": f"same daily temperature path shifted by -{config['estimand']['counterfactual_shift_c']} C on every day",
        "estimand": "sum_t [E(Y_t | factual temperature history) - E(Y_t | counterfactual temperature history)]",
        "poisson_pseudo_likelihood": True,
        "model_based_and_hac_intervals_are_approximate": True,
        "no_exact_coverage_claim": True,
        "design": {
            "temperature_basis": "centered natural cubic spline with two columns; factual min/median/max knots; linear continuation outside boundaries",
            "lag_basis": f"three-column natural cubic cardinal basis over physical days 0 through {base['max_lag']}; lag components are not physical lag blocks",
            "time_spline": f"centered natural cubic spline with {base['time_df']} columns over {design_cfg['study_years']} prespecified years ({design_cfg['primary_time_df_per_year']} df/year)",
            "weekday": "six centered day-of-week indicators",
            "design_columns": int(base["X"].shape[1]),
            "design_rank": int(base["design_meta"]["rank"]),
            "condition_number_unweighted_design": float(base["design_meta"]["singular_value_condition_number"]),
            "condition_number_weighted_information": float(observed["weighted_information_condition_number"]),
        },
        "fit": {
            "family": "Poisson log-link",
            "method": "line-searched Fisher scoring / IRLS",
            "converged": bool(observed["converged"]),
            "convergence_reason": observed["convergence_reason"],
            "iterations": observed["iterations"],
            "max_abs_step": observed["last_max_abs_step"],
            "max_abs_score": observed["max_abs_score"],
            "log_likelihood": observed["loglik"],
            "pearson_dispersion_diagnostic": float(np.sum((base["y"] - observed["mu"]) ** 2 / observed["mu"]) / max(1, observed["n"] - observed["p"])),
        },
        "simulation_dgp": {
            "model_compatible": "independent Poisson outcomes generated from the observed-data primary fitted marginal means",
            "smooth_omitted_interaction": (
                "adds the prespecified seasonal temperature-response term to the factual and counterfactual log means; "
                "its amplitude is a scenario setting, not an empirical error bound"
            ),
            "correlated_overdispersed_ar1": {
                "latent_process": "stationary Gaussian AR(1)",
                "phi": float(config["simulation"]["correlated_overdispersed"]["phi"]),
                "estimated_lognormal_sigma2": float(lognormal_sigma2),
                "variance_estimator": "max(log(1 + max(sum((y-mu)^2)-sum(mu), 0)/sum(mu^2)), 0)",
                "marginal_mean_preserved": True,
            },
        },
        "intervals": interval_rows,
        "interpretation": "These estimates condition on this daily aggregate panel, the point temperature proxy, the stated spline model, and the fixed -1 C path.",
    }
    atomic_json(RESULTS_DIR / "real_observed_model.json", primary_report)
    _write_design_artifact(RESULTS_DIR / "real_design.npz", base, observed)
    sensitivity = _sensitivity_table(config, panel)
    atomic_csv(RESULTS_DIR / "real_sensitivity.csv", sensitivity)

    simulations, summary, paired = _run_simulations(config, base, observed, panel, mode, workers, pilot_reps)
    atomic_csv(RESULTS_DIR / f"real_{mode}_replications.csv", simulations)
    atomic_csv(RESULTS_DIR / f"real_{mode}_summary.csv", summary)
    atomic_csv(RESULTS_DIR / f"real_{mode}_paired_differences.csv", paired)

    repro = {
        "mode": mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "input": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "sha256": panel_sha,
            "selected_state": str(design_cfg["state_code"]).zfill(2),
            "selected_rows": int(len(panel)),
            "selected_period": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        },
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(Path(__file__)),
        "workers": min(int(workers), int(config["execution"]["max_workers"])),
        "blas_threads_per_worker": 1,
        "replications": int(len(simulations)),
        "replications_per_scenario": int(
            config["simulation"][f"{mode}_reps_per_scenario"] if pilot_reps is None else pilot_reps
        ),
        "pilot_replication_override": int(pilot_reps) if pilot_reps is not None else None,
        "converged_replications": int(simulations["converged"].sum()),
        "failed_replications": int((~simulations["converged"]).sum()),
        "primary_fit_converged": bool(observed["converged"]),
        "primary_fit_iterations": int(observed["iterations"]),
        "primary_fit_condition_number": float(observed["weighted_information_condition_number"]),
        "outputs": [
            "results/real_observed_model.json",
            "results/real_design.npz",
            "results/real_sensitivity.csv",
            f"results/real_{mode}_replications.csv",
            f"results/real_{mode}_summary.csv",
            f"results/real_{mode}_paired_differences.csv",
        ],
        "limits": [
            "HAC intervals use finite-bandwidth Bartlett score covariance and have no exact finite-sample coverage guarantee.",
            "Simulation scenarios are conditional on one fixed design; the smooth interaction scale is prespecified and is not an empirical temperature-error bound.",
            "No population denominator is available in the aggregate panel; the outcome is daily state death count.",
        ],
    }
    atomic_json(RESULTS_DIR / f"real_{mode}_reproducibility.json", repro)
    print(json.dumps({
        "mode": mode,
        "elapsed_seconds": repro["elapsed_seconds"],
        "primary_converged": repro["primary_fit_converged"],
        "primary_iterations": repro["primary_fit_iterations"],
        "replications": repro["replications"],
        "converged_replications": repro["converged_replications"],
        "failed_replications": repro["failed_replications"],
        "primary_A_hat": primary_report["intervals"][0]["A_hat_deaths"],
        "condition_number": repro["primary_fit_condition_number"],
    }, indent=2))
    return repro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot-reps", type=int, help="optional short pilot override; formal counts remain config-locked")
    args = parser.parse_args()
    try:
        run(args.mode, args.workers, args.pilot_reps)
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
