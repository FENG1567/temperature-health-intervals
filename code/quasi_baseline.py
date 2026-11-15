#!/usr/bin/env python3
"""Replay approved fixed-design outcomes with a quasi-Poisson scale baseline.

This post-hoc comparator reuses the exact formal seeds and data-generating
means from real_panel_audit.py. The original Poisson and HAC results remain
unchanged; only the model-based covariance is rescaled by the replicate-level
Pearson dispersion estimate, without flooring that estimate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
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

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


PROJECT_ROOT = source_root()
_WORKER: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def _case_root(case: str) -> Path:
    return PROJECT_ROOT / "city_case" if case == "city" else PROJECT_ROOT


def _load_engine(case: str):
    root = _case_root(case)
    source = root / "code" / "real_panel_audit.py"
    spec = importlib.util.spec_from_file_location(f"real_panel_engine_{case}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load approved audit engine at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return root, module


def _prepare_case(case: str) -> dict[str, Any]:
    root, engine = _load_engine(case)
    config = engine.read_config()
    if config.get("protocol_status") != "approved":
        raise PermissionError(f"{case} config is not approved.")
    formal_path = root / "results" / "real_formal_replications.csv"
    repro_path = root / "results" / "real_formal_reproducibility.json"
    if not formal_path.exists() or not repro_path.exists():
        raise FileNotFoundError(f"Approved formal outputs are missing for {case}.")
    formal = pd.read_csv(formal_path)
    formal_repro = json.loads(repro_path.read_text(encoding="utf-8"))
    if formal_repro["mode"] != "formal" or int(formal_repro["replications_per_scenario"]) != 1000:
        raise ValueError(f"Formal replication manifest is not the expected 1000-per-scenario run for {case}.")
    code_path = root / "code" / "real_panel_audit.py"
    # Portable IO wrapper changes the source hash; immutable inputs/config remain checked below.
    if _sha256(engine.CONFIG_PATH) != formal_repro["config_sha256"]:
        raise ValueError(f"{case} config hash differs from the approved formal run.")
    if _sha256(engine.DATA_PATH) != formal_repro["input"]["sha256"]:
        raise ValueError(f"{case} input hash differs from the approved formal run.")

    data = pd.read_parquet(engine.DATA_PATH)
    data["state"] = data["state"].astype(str).str.zfill(2)
    data["date"] = pd.to_datetime(data["date"])
    design_cfg = config["design"]
    panel = data.loc[
        data["state"].eq(str(design_cfg["state_code"]).zfill(2))
        & data["date"].dt.year.between(int(design_cfg["start_year"]), int(design_cfg["end_year"]))
    ].sort_values("date").reset_index(drop=True)
    design = engine.build_design(
        panel,
        max_lag=int(design_cfg["primary_max_lag"]),
        time_df=int(design_cfg["primary_time_df_per_year"]) * int(design_cfg["study_years"]),
        trim_days=int(design_cfg["common_trim_days"]),
        cf_shift_c=float(config["estimand"]["counterfactual_shift_c"]),
    )
    fit_cfg = config["fit"]
    inference = config["inference"]
    observed = engine.fit_poisson_irls(
        design["X"], design["y"], design["offset"], Xcf=design["Xcf"],
        hac_bandwidths=tuple(int(x) for x in inference["hac_bandwidths"]),
        max_iterations=int(fit_cfg["max_iterations"]),
        tolerance=float(fit_cfg["tolerance_max_abs_step"]),
    )
    if not observed["converged"]:
        raise RuntimeError(f"Observed {case} fit failed during quasi baseline replay.")

    means_compatible = np.asarray(observed["mu"], dtype=float)
    cf_means_compatible = np.asarray(observed["mu_cf"], dtype=float)
    row_indices = np.asarray(design["row_indices"], dtype=int)
    dates = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d").to_numpy()[row_indices]
    temperature = panel["tmean"].to_numpy(dtype=float)[row_indices]
    counterfactual_temperature = temperature - float(design["cf_shift_c"])
    interaction_amplitude = float(config["simulation"]["omitted_interaction"]["amplitude"])
    h_f = engine._interaction_component(dates, temperature, interaction_amplitude)
    h_cf = engine._interaction_component(dates, counterfactual_temperature, interaction_amplitude)
    means_omitted = means_compatible * np.exp(h_f)
    cf_means_omitted = cf_means_compatible * np.exp(h_cf)
    sigma2 = engine._lognormal_variance_from_residuals(design["y"], means_compatible)
    phi = float(config["simulation"]["correlated_overdispersed"]["phi"])
    scenario_means = {
        "model_compatible": means_compatible,
        "smooth_omitted_interaction": means_omitted,
        "correlated_overdispersed_ar1": means_compatible,
    }
    scenario_truth = {
        "model_compatible": float(np.sum(means_compatible - cf_means_compatible)),
        "smooth_omitted_interaction": float(np.sum(means_omitted - cf_means_omitted)),
        "correlated_overdispersed_ar1": float(np.sum(means_compatible - cf_means_compatible)),
    }
    sim = config["simulation"]
    scenario_ids = {item["name"]: int(item["seed_id"]) for item in sim["scenarios"]}
    if set(scenario_ids) != set(scenario_means):
        raise ValueError(f"Scenario IDs and DGP means differ for {case}.")

    # Independently report the observed-data quasi-Poisson interval as a
    # diagnostic. The formal comparator below estimates dispersion per draw.
    observed_phi = float(
        np.sum((design["y"] - observed["mu"]) ** 2 / observed["mu"])
        / (observed["n"] - observed["p"])
    )
    zcrit = float(norm.ppf(1.0 - float(inference["alpha"]) / 2.0))
    observed_se_model = float(observed["se"]["model"])
    observed_se_quasi = math.sqrt(observed_phi) * observed_se_model
    observed_quasi = {
        "phi_hat_observed": observed_phi,
        "A_hat": float(observed["A_hat"]),
        "se_model": observed_se_model,
        "se_quasi": observed_se_quasi,
        "normal_critical_value": zcrit,
        "interval_level": float(1.0 - inference["alpha"]),
        "interval": [
            float(observed["A_hat"] - zcrit * observed_se_quasi),
            float(observed["A_hat"] + zcrit * observed_se_quasi),
        ],
        "status": "post-hoc quasi-Poisson scale diagnostic; primary observed model output remains frozen",
    }
    return {
        "case": case,
        "root": root,
        "engine": engine,
        "config": config,
        "formal": formal,
        "formal_repro": formal_repro,
        "design": design,
        "observed_fit": observed,
        "scenario_means": scenario_means,
        "scenario_truth": scenario_truth,
        "scenario_ids": scenario_ids,
        "sigma2": float(sigma2),
        "phi_ar1": phi,
        "observed_quasi": observed_quasi,
        "zcrit": zcrit,
    }


def _init_worker(state: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = state


def _draw_y(state: dict[str, Any], scenario: str, replicate: int, seed_parts: list[int]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    means = state["scenario_means"][scenario]
    if scenario != "correlated_overdispersed_ar1":
        return rng.poisson(means).astype(np.int64)
    sigma2 = float(state["sigma2"])
    phi = float(state["phi_ar1"])
    if sigma2 <= 0:
        latent = np.zeros(len(means), dtype=float)
    else:
        latent = np.empty(len(means), dtype=float)
        latent[0] = rng.normal(0.0, math.sqrt(sigma2))
        innovation_sd = math.sqrt(sigma2 * (1.0 - phi * phi))
        for index in range(1, len(means)):
            latent[index] = phi * latent[index - 1] + rng.normal(0.0, innovation_sd)
    return rng.poisson(means * np.exp(latent - 0.5 * sigma2)).astype(np.int64)


def _one_replay(task: tuple[str, int]) -> dict[str, Any]:
    scenario, replicate = task
    if _WORKER is None:
        raise RuntimeError("Quasi replay worker state was not initialized.")
    state = _WORKER
    case = state["case_state"]
    config = case["config"]
    sim = config["simulation"]
    formal_repro = case["formal_repro"]
    seed_parts = [
        int(sim["base_seed"]),
        int(sim["mode_ids"]["formal"]),
        int(case["scenario_ids"][scenario]),
        int(replicate),
    ]
    y = _draw_y(case, scenario, replicate, seed_parts)
    y_hash = hashlib.sha256(y.tobytes(order="C")).hexdigest()
    original = state["original_rows"][(scenario, int(replicate))]
    try:
        fit_cfg = config["fit"]
        fit = case["engine"].fit_poisson_irls(
            case["design"]["X"], y.astype(float), case["design"]["offset"],
            Xcf=case["design"]["Xcf"], hac_bandwidths=(14, 28),
            max_iterations=int(fit_cfg["max_iterations"]),
            tolerance=float(fit_cfg["tolerance_max_abs_step"]),
        )
        if not fit["converged"]:
            raise RuntimeError(fit["convergence_reason"])
        n, p = int(fit["n"]), int(fit["p"])
        # Canonical quasi-Poisson scale estimate, with no lower bound or cap.
        phi_hat = float(np.sum((y - fit["mu"]) ** 2 / fit["mu"]) / (n - p))
        se_model = float(fit["se"]["model"])
        se_quasi = math.sqrt(phi_hat) * se_model
        a_hat = float(fit["A_hat"])
        a_true = float(original["A_true"])
        lo = a_hat - float(case["zcrit"]) * se_quasi
        hi = a_hat + float(case["zcrit"]) * se_quasi
        a_original = float(original["A_hat"])
        se_original = float(original["se_model"])
        rel_a = abs(a_hat - a_original) / max(1.0, abs(a_original))
        rel_se = abs(se_model - se_original) / max(1.0, abs(se_original))
        return {
            "scenario": scenario,
            "replicate": int(replicate),
            "seed_entropy": json.dumps(seed_parts, separators=(",", ":")),
            "outcome_sha256": y_hash,
            "converged": True,
            "failure_reason": "",
            "iterations": int(fit["iterations"]),
            "phi_hat": phi_hat,
            "A_hat": a_hat,
            "A_true": a_true,
            "se_model_replay": se_model,
            "se_model_original": se_original,
            "A_hat_original": a_original,
            "relative_A_hat_difference": float(rel_a),
            "relative_se_model_difference": float(rel_se),
            "match_A_hat_1e_8": bool(rel_a <= 1e-8),
            "match_se_model_1e_8": bool(rel_se <= 1e-8),
            "se_quasi": float(se_quasi),
            "interval_lo_quasi": float(lo),
            "interval_hi_quasi": float(hi),
            "cover_quasi": bool(lo <= a_true <= hi),
            "width_quasi": float(hi - lo),
            "cover_model": bool(original["cover_model"]),
            "width_model": float(original["width_model"]),
            "cover_hac14": bool(original["cover_hac14"]),
            "width_hac14": float(original["width_hac14"]),
            "cover_hac28": bool(original["cover_hac28"]),
            "width_hac28": float(original["width_hac28"]),
        }
    except Exception as exc:
        return {
            "scenario": scenario,
            "replicate": int(replicate),
            "seed_entropy": json.dumps(seed_parts, separators=(",", ":")),
            "outcome_sha256": y_hash,
            "converged": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "iterations": None,
            "phi_hat": None,
            "A_hat": None,
            "A_true": float(original["A_true"]),
            "se_model_replay": None,
            "se_model_original": float(original["se_model"]),
            "A_hat_original": float(original["A_hat"]),
            "relative_A_hat_difference": None,
            "relative_se_model_difference": None,
            "match_A_hat_1e_8": False,
            "match_se_model_1e_8": False,
            "se_quasi": None,
            "interval_lo_quasi": None,
            "interval_hi_quasi": None,
            "cover_quasi": False,
            "width_quasi": None,
            "cover_model": bool(original["cover_model"]),
            "width_model": float(original["width_model"]),
            "cover_hac14": bool(original["cover_hac14"]),
            "width_hac14": float(original["width_hac14"]),
            "cover_hac28": bool(original["cover_hac28"]),
            "width_hac28": float(original["width_hac28"]),
        }


def _summary_rows(results: pd.DataFrame, case_state: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    paired = []
    for scenario in case_state["scenario_means"]:
        group = results.loc[results["scenario"].eq(scenario)]
        success = group.loc[group["converged"].astype(bool)]
        covered = int(group["cover_quasi"].sum())
        n_planned = int(len(group))
        n_success = int(len(success))
        cov_plan = covered / n_planned if n_planned else None
        cov_success = int(success["cover_quasi"].sum()) / n_success if n_success else None
        errors = success["A_hat"].astype(float) - success["A_true"].astype(float)
        empirical_sd = float(errors.std(ddof=1)) if len(errors) > 1 else None
        mean_se = float(success["se_quasi"].mean()) if n_success else None
        summaries.append({
            "case": case_state["case"],
            "scenario": scenario,
            "method": "quasi_poisson_pearson_scale",
            "A_true": float(case_state["scenario_truth"][scenario]),
            "n_planned": n_planned,
            "n_converged": n_success,
            "n_failed": int(n_planned - n_success),
            "covered_planned_failures_uncovered": covered,
            "coverage_planned": float(cov_plan) if cov_plan is not None else None,
            "coverage_mcse_planned": (
                float(math.sqrt(cov_plan * (1.0 - cov_plan) / n_planned)) if cov_plan is not None else None
            ),
            "covered_converged": int(success["cover_quasi"].sum()),
            "coverage_converged_only": float(cov_success) if cov_success is not None else None,
            "coverage_mcse_converged_only": (
                float(math.sqrt(cov_success * (1.0 - cov_success) / n_success))
                if cov_success is not None else None
            ),
            "mean_phi_hat": float(success["phi_hat"].mean()) if n_success else None,
            "mean_se_quasi": mean_se,
            "empirical_sd_error": empirical_sd,
            "mean_se_quasi_over_empirical_sd": (
                float(mean_se / empirical_sd) if mean_se is not None and empirical_sd is not None and empirical_sd > 0 else None
            ),
            "mean_interval_width_converged": float(success["width_quasi"].mean()) if n_success else None,
            "mean_bias_converged": float(errors.mean()) if n_success else None,
            "max_relative_A_hat_difference": float(success["relative_A_hat_difference"].max()) if n_success else None,
            "max_relative_se_model_difference": float(success["relative_se_model_difference"].max()) if n_success else None,
            "all_replay_checks_pass": bool(
                n_success == n_planned
                and success["match_A_hat_1e_8"].all()
                and success["match_se_model_1e_8"].all()
            ),
        })
        for method in ("model", "hac14", "hac28"):
            old_cover = group[f"cover_{method}"].astype(float)
            new_cover = group["cover_quasi"].astype(float)
            pair_success = group.loc[group["converged"].astype(bool)]
            width_diff = (
                pd.to_numeric(pair_success["width_quasi"], errors="coerce")
                - pd.to_numeric(pair_success[f"width_{method}"], errors="coerce")
            ).dropna()
            cover_diff = new_cover - old_cover
            paired.append({
                "case": case_state["case"],
                "scenario": scenario,
                "method_1": "quasi",
                "method_2": method,
                "n_planned": n_planned,
                "mean_paired_coverage_difference": float(cover_diff.mean()) if n_planned else None,
                "mcse_paired_coverage_difference": (
                    float(cover_diff.std(ddof=1) / math.sqrt(n_planned)) if n_planned > 1 else None
                ),
                "n_converged_width_pairs": int(len(width_diff)),
                "mean_paired_width_difference_quasi_minus_comparator": (
                    float(width_diff.mean()) if len(width_diff) else None
                ),
                "mcse_paired_width_difference": (
                    float(width_diff.std(ddof=1) / math.sqrt(len(width_diff))) if len(width_diff) > 1 else None
                ),
            })
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def run(case: str, workers: int = 4) -> dict[str, Any]:
    started = time.perf_counter()
    if case not in {"state", "city"}:
        raise ValueError("case must be state or city.")
    if workers < 1:
        raise ValueError("workers must be at least one.")
    state = _prepare_case(case)
    root = state["root"]
    formal = state["formal"].copy()
    expected_replications = 3 * 1000
    if len(formal) != expected_replications:
        raise ValueError(f"Expected {expected_replications} formal rows, found {len(formal)}.")
    sim = state["config"]["simulation"]
    expected_ids = state["scenario_ids"]
    if set(formal["scenario"].unique()) != set(expected_ids):
        raise ValueError("Formal output scenario names differ from approved config.")
    original_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in formal.to_dict(orient="records"):
        scenario = str(row["scenario"])
        replicate = int(row["replicate"])
        expected_seed = [
            int(sim["base_seed"]),
            int(sim["mode_ids"]["formal"]),
            int(expected_ids[scenario]),
            replicate,
        ]
        stored_seed = json.loads(row["seed_entropy"])
        if stored_seed != expected_seed:
            raise ValueError(f"Formal seed mismatch for {scenario} replicate {replicate}: {stored_seed} != {expected_seed}")
        original_rows[(scenario, replicate)] = row
    if len(original_rows) != expected_replications:
        raise ValueError("Formal replication identifiers are not unique.")

    worker_state = {
        "case_state": state,
        "original_rows": original_rows,
    }
    tasks = [(scenario, replicate) for scenario in state["scenario_means"] for replicate in range(1000)]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, 4),
            initializer=_init_worker,
            initargs=(worker_state,),
        ) as pool:
            rows = list(pool.map(_one_replay, tasks, chunksize=1))
    else:
        _init_worker(worker_state)
        rows = [_one_replay(task) for task in tasks]
    results = pd.DataFrame(rows)
    summary, paired = _summary_rows(results, state)
    outdir = output_path(root / "results")
    rep_path = outdir / "quasi_formal_replications.csv"
    summary_path = outdir / "quasi_formal_summary.csv"
    paired_path = outdir / "quasi_formal_paired_differences.csv"
    repro_path = outdir / "quasi_formal_reproducibility.json"
    _write_csv(rep_path, results)
    _write_csv(summary_path, summary)
    _write_csv(paired_path, paired)

    all_checks = bool(
        len(results) == expected_replications
        and results["converged"].all()
        and results["match_A_hat_1e_8"].all()
        and results["match_se_model_1e_8"].all()
        and results["outcome_sha256"].nunique() == expected_replications
    )
    repro = {
        "case": case,
        "case_id": state["config"].get("case_id", "real_daily_state_35_v1"),
        "mode": "formal_quasi_baseline_replay",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "project_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "workers": min(workers, 4),
        "blas_threads_per_worker": 1,
        "input_sha256": state["formal_repro"]["input"]["sha256"],
        "approved_config_sha256": state["formal_repro"]["config_sha256"],
        "engine_code_sha256": state["formal_repro"]["code_sha256"],
        "quasi_script_sha256": _sha256(Path(__file__)),
        "formal_reproducibility_sha256": _sha256(root / "results" / "real_formal_reproducibility.json"),
        "seed_replay": {
            "seed_rule": "NumPy SeedSequence([base_seed, formal_mode_id, scenario_seed_id, replicate_index])",
            "seed_entropy_matched_formal_csv": True,
            "unique_replayed_outcome_hashes": int(results["outcome_sha256"].nunique()),
            "planned_replicates": expected_replications,
            "all_replay_A_hat_and_model_SE_match_within_relative_1e_8": bool(
                results["match_A_hat_1e_8"].all() and results["match_se_model_1e_8"].all()
            ),
            "max_relative_A_hat_difference": float(results["relative_A_hat_difference"].max()),
            "max_relative_model_SE_difference": float(results["relative_se_model_difference"].max()),
        },
        "quasi_poisson": {
            "dispersion": "phi_hat = sum((y - mu_hat)^2 / mu_hat) / (n - p)",
            "scale_floor": False,
            "standard_error": "sqrt(phi_hat) * original Poisson information-based SE(A)",
            "interval": "two-sided normal Wald interval at the approved alpha",
            "observed_data_diagnostic": state["observed_quasi"],
        },
        "all_mechanical_checks_pass": all_checks,
        "outputs": [str(rep_path.relative_to(root)), str(summary_path.relative_to(root)), str(paired_path.relative_to(root))],
        "interpretation": (
            "Exploratory quasi-Poisson baseline added after the original formal "
            "summary; original primary Poisson/HAC files were not modified. "
            "This rescaling is not a finite-sample coverage theorem."
        ),
    }
    _write_json(repro_path, repro)
    print(json.dumps({
        "case": case,
        "elapsed_seconds": repro["elapsed_seconds"],
        "n_replicates": len(results),
        "n_converged": int(results["converged"].sum()),
        "all_replay_checks_pass": all_checks,
        "max_relative_A_hat_difference": repro["seed_replay"]["max_relative_A_hat_difference"],
        "max_relative_model_SE_difference": repro["seed_replay"]["max_relative_model_SE_difference"],
        "observed_phi_hat": state["observed_quasi"]["phi_hat_observed"],
    }, indent=2))
    if not all_checks:
        raise RuntimeError(f"Quasi baseline replay mechanical checks failed for {case}.")
    return repro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("state", "city"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        run(args.case, args.workers)
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
