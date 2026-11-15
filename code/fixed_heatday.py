"""Fixed-design outcome-resampling adapter for the heat-day DGP."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
import scipy

ROOT = source_root()
CONFIG = ROOT / "inputs" / "fixed_heatday_config.json"
SNAPSHOT = ROOT / "inputs" / "heatday_snapshot"
ENGINE_FILE = ROOT / "code" / "fixed_heatday_engine.py"
CORE_HD = SNAPSHOT / "core_hd.py"
CORE_SIM = SNAPSHOT / "core_sim.py"
RUN_FAIL_HD = SNAPSHOT / "run_fail_hd.py"
CORE_FAIL_HD = SNAPSHOT / "core_fail_hd.py"

_engine_spec = importlib.util.spec_from_file_location("fixed_heatday_engine_snapshot", ENGINE_FILE)
if _engine_spec is None or _engine_spec.loader is None:
    raise RuntimeError(f"frozen engine snapshot not found: {ENGINE_FILE}")
engine = importlib.util.module_from_spec(_engine_spec)
sys.modules[_engine_spec.name] = engine
_engine_spec.loader.exec_module(engine)

# core_hd imports its own pinned core_sim helpers from this snapshot directory.
sys.path.insert(0, str(SNAPSHOT))
_hd_spec = importlib.util.spec_from_file_location("pinned_core_hd", CORE_HD)
if _hd_spec is None or _hd_spec.loader is None:
    raise RuntimeError(f"heat-day source snapshot not found: {CORE_HD}")
heatday = importlib.util.module_from_spec(_hd_spec)
sys.modules[_hd_spec.name] = heatday
_hd_spec.loader.exec_module(heatday)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    expected = {
        ENGINE_FILE: cfg["frozen_engine"]["sha256"],
        CORE_HD: cfg["source"]["core_hd_sha256"],
        CORE_SIM: cfg["source"]["core_sim_sha256"],
        RUN_FAIL_HD: cfg["source"]["run_fail_hd_sha256"],
        CORE_FAIL_HD: cfg["source"]["core_fail_hd_sha256"],
    }
    for path, digest in expected.items():
        if sha(path).lower() != str(digest).lower():
            raise RuntimeError(f"pinned heat-day input hash mismatch: {path}={sha(path)}")
    return cfg


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def make_design(cfg: dict[str, Any], sc: dict[str, Any], seed: int) -> dict[str, Any]:
    d = cfg["design"]
    R, T = int(d["R"]), int(d["T"])
    if int(d["n_c"]) != 1:
        raise ValueError("fixed heat-day protocol requires n_c=1")
    sim_cfg = heatday.Cfg(n_c=1, R=R, T=T, theta=float(d["theta"]),
        rate_lo=float(d["rate_lo"]), rate_hi=float(d["rate_hi"]), bias=0.0,
        common_bias=float(sc["common_bias"]), misspec=False, label=sc["name"])
    dat = heatday.simulate_hd(sim_cfg, int(seed))
    h = np.asarray(dat["H"][0], float).reshape(-1, 2)
    hcf = np.asarray(dat["Hcf"][0], float).reshape(-1, 2)
    region = np.repeat(np.arange(R), T)
    month = np.tile(np.arange(T) % 12, R)
    trend = np.tile(np.arange(T) / 12.0, R)
    year = np.tile(np.arange(T) // 12, R)
    off = np.log(np.repeat(dat["pop"][0], T))
    knots = np.asarray(d["fixed_knots"], float)
    X = heatday.build_X(h, region, month, trend, knots, R)
    Xcf = heatday.cf_X(X, hcf, knots, shift=0.0)
    mu = np.asarray(dat["mu"][0], float).reshape(-1).copy()
    mucf = np.asarray(dat["mucf"][0], float).reshape(-1).copy()
    beta = np.full(X.shape[1], np.nan)
    source_projection_rmse = float("nan")
    pc_fit_ok = None
    pc_beta_error = float("nan")
    pc_mu_error = float("nan")
    if sc["kind"] == "exact_model":
        lm = np.log(np.maximum(mu, 1e-12)) - off
        beta = np.linalg.lstsq(X, lm, rcond=None)[0]
        source_projection_rmse = float(np.sqrt(np.mean((lm - X @ beta) ** 2)))
        mu = np.exp(np.clip(X @ beta + off, -30, 30))
        mucf = np.exp(np.clip(Xcf @ beta + off, -30, 30))
        pc = engine.fit_tw(mu, X, off, region, year, cfg)
        pc_beta_error = float(np.max(np.abs(pc["beta"] - beta)))
        pc_mu = np.exp(np.clip(X @ pc["beta"] + off, -30, 30))
        pc_mu_error = float(np.max(np.abs(pc_mu - mu)))
        pc_fit_ok = bool(pc["conv"] and pc["rank"] == X.shape[1]
            and np.isfinite(pc_beta_error) and pc_beta_error < 1e-6
            and np.isfinite(pc_mu_error) and pc_mu_error < 1e-6)
        if not pc_fit_ok:
            raise RuntimeError(f"heat-day exact-model positive control failed for seed={seed}: "
                f"converged={pc['conv']}, rank={pc['rank']}/{X.shape[1]}, "
                f"beta_error={pc_beta_error}, mu_error={pc_mu_error}")
    truth = float(np.sum(mu - mucf))
    z = {"scenario": sc["name"], "scenario_id": int(sc["id"]), "design_seed": int(seed),
        "theta": float(d["theta"]), "truth": truth, "cfg": sim_cfg,
        "X": X, "Xcf": Xcf, "off": off, "region": region, "year": year,
        "mu": mu, "mucf": mucf, "beta": beta, "knots": knots,
        "H": dat["H"][0], "Hcf": dat["Hcf"][0], "pop": dat["pop"][0], "thr": dat["thr"][0],
        "source_projection_rmse": source_projection_rmse, "positive_control_fit_ok": pc_fit_ok,
        "positive_control_beta_max_abs_error": pc_beta_error,
        "positive_control_mu_max_abs_error": pc_mu_error,
        "mean_H": float(dat["mean_H"]), "frac_hot": float(dat["frac_hot"])}
    path = run_root() / "results" / "heatday_designs" / f"heatday_design_{sc['name']}_seed{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, X=X, Xcf=Xcf, offset=off, region=region, year=year,
            mu=mu, mucf=mucf, beta_fixed=beta, knots=knots, H=dat["H"][0], Hcf=dat["Hcf"][0],
            pop=dat["pop"][0], threshold=dat["thr"][0], A_true=truth, theta=sim_cfg.theta,
            source_projection_rmse=source_projection_rmse,
            positive_control_fit_ok=(bool(pc_fit_ok) if pc_fit_ok is not None else np.nan),
            positive_control_beta_max_abs_error=pc_beta_error, positive_control_mu_max_abs_error=pc_mu_error,
            metadata_json=np.asarray(json.dumps({"scenario": sc["name"], "seed": seed, "R": R, "T": T,
                "theta": sim_cfg.theta, "knots": knots.tolist(), "common_bias": sc["common_bias"],
                "counterfactual_bias": 0.0, "source_sha256": sha(CORE_HD),
                "core_sim_sha256": sha(CORE_SIM), "truth": truth}, ensure_ascii=False)))
    tmp.replace(path)
    z["design_path"] = str(path)
    z["design_sha256"] = sha(path)
    return z


def draw_y(D: dict[str, Any], cfg: dict[str, Any], scid: int, mode: str, stage: str, rep: int) -> np.ndarray:
    return engine.draw_y(D, cfg, scid, mode, stage, rep)


def summarize_panel(rows: list[dict[str, Any]], method: str, scenario: str, seed: int,
                    eval_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return engine.summarize(rows, method, scenario, seed, eval_rows=eval_rows)


def run_panel(task: tuple[str, int, str]) -> dict[str, Any]:
    scname, seed, mode = task
    cfg = read_config()
    sc = next(s for s in cfg["design"]["scenarios"] if s["name"] == scname)
    D = make_design(cfg, sc, seed)
    mode_id = int(cfg["sampling"]["mode_ids"][mode])
    counts = cfg["sampling"][f"{mode}_reps_per_panel"]
    stages: dict[str, list[dict[str, Any]]] = {}
    for stage in ("bias_train", "scale_calibration", "test"):
        stage_rows = []
        for i in range(int(counts[stage])):
            y = draw_y(D, cfg, int(sc["id"]), mode, stage, i)
            r = engine.fit_rep(y, D, cfg)
            r.update({"scenario": scname, "scenario_id": int(sc["id"]), "design_seed": seed,
                "mode": mode, "mode_id": mode_id, "stage": stage, "replicate": i})
            stage_rows.append(r)
        stages[stage] = stage_rows
    tr, cal, te = stages["bias_train"], stages["scale_calibration"], stages["test"]
    train_errors = np.array([r["error"] for r in tr if r["fit_ok"] and np.isfinite(r["error"])])
    bias = float(train_errors.mean()) if len(train_errors) else float("nan")
    cal_valid = [r for r in cal if r["fit_ok"] and np.isfinite(r["error"])
        and np.isfinite(r["h_corrected"]) and r["h_corrected"] > 0]
    raw_scores = np.array([abs(r["error"]) / r["h_corrected"] for r in cal_valid], float)
    centered_scores = np.array([abs(r["error"] - bias) / r["h_corrected"]
        for r in cal_valid if np.isfinite(bias)], float)
    alpha = float(cfg["sampling"]["alpha"])
    qraw = engine.qfinite(raw_scores, alpha)
    qb = engine.qfinite(centered_scores, alpha)
    shift = float(cfg["shared_additive_shift"])
    bias_common = float(bias + shift) if np.isfinite(bias) else float("nan")
    raw_common_scores = np.array([abs(r["error"] + shift) / r["h_corrected"] for r in cal_valid], float)
    centered_common_scores = np.array([abs(r["error"] + shift - bias_common) / r["h_corrected"]
        for r in cal_valid if np.isfinite(bias_common)], float)
    qraw_common = engine.qfinite(raw_common_scores, alpha)
    qb_common = engine.qfinite(centered_common_scores, alpha)
    max_centered_score_change = max([abs((abs(r["error"] + shift - bias_common)
        - abs(r["error"] - bias)) / r["h_corrected"]) for r in cal_valid
        if np.isfinite(bias) and np.isfinite(bias_common)] or [0.0])
    year_n = min(int(cfg["year_split_diagnostic"][f"{mode}_test_reps_per_panel"]), len(te))
    if not cfg["year_split_diagnostic"].get("enabled", False):
        year_n = 0
    for i, r in enumerate(te):
        a, truth = r["A_hat"], r["A_true"]
        r["A_hat_common_bias"] = a + shift if np.isfinite(a) else float("nan")
        r["error_common_bias"] = r["error"] + shift if np.isfinite(r["error"]) else float("nan")
        r["shared_additive_shift"] = shift
        if i < year_n and r["fit_ok"]:
            y = draw_y(D, cfg, int(sc["id"]), mode, "test", i)
            yr = engine.yearhalf_q_diagnostic(y, D, cfg, int(sc["id"]), mode, i)
            r["yearhalf_q"] = yr["q"]
            r["yearhalf_splits_requested"] = yr["n_split_requested"]
            r["yearhalf_splits_valid"] = yr["n_split_valid"]
            r["yearhalf_splits_failed"] = yr["n_split_failed"]
        else:
            r["yearhalf_q"] = float("nan")
            r["yearhalf_splits_requested"] = 0
            r["yearhalf_splits_valid"] = 0
            r["yearhalf_splits_failed"] = 0
        specs = {
            "delta_corrected_cr1_tw": (a, r["h_corrected"], r["tcrit_exact"]),
            "delta_corrected_common_cr1_tw": (a, r["h_common_cr1"], r["tcrit_exact"]),
            "delta_legacy_covariance": (a, r["h_legacy"], r["tcrit_exact"]),
            "delta_pinned_source": (a, r["h_legacy"], r["tcrit_source_tq"]),
            "delta_region_cr1_context": (a, r["h_region_cr1"], r["tcrit_region"]),
            "delta_independent_HC1_positive_control": (a, r["h_hc1"], r["tcrit_hc1"]),
            "independent_scale_calibrated": (a, r["h_corrected"], qraw),
            "truth_trained_bias_centered_split_calibrated": (a - bias, r["h_corrected"], qb),
            "estimator_only_common_bias": (a + shift, r["h_corrected"], r["tcrit_exact"]),
            "estimator_only_common_bias_bias_centered": (a + shift - bias_common, r["h_corrected"], qb_common),
        }
        if i < year_n:
            specs["year_half_split_q_diagnostic"] = (a, r["h_corrected"], r["yearhalf_q"])
        for method, (center, h, q) in specs.items():
            lo, hi, cover, width = engine.interval(center, h, q, truth) if r["fit_ok"] else (
                float("nan"), float("nan"), 0, float("nan"))
            valid_interval = not (np.isnan(lo) or np.isnan(hi) or np.isnan(width))
            r["lo__" + method] = lo
            r["hi__" + method] = hi
            r["cover__" + method] = cover
            r["width__" + method] = width
            r["interval_valid__" + method] = bool(valid_interval)
    equiv_error = max([abs(((r["A_hat"] + shift) - (r["A_true"] + shift)) - r["error"])
        for r in te if np.isfinite(r["error"])] or [0.0])
    common_centered_maxdiff = max([abs((r["A_hat"] + shift - bias_common) - (r["A_hat"] - bias))
        for r in te if np.isfinite(r["A_hat"]) and np.isfinite(bias) and np.isfinite(bias_common)] or [0.0])
    methods = ["delta_corrected_cr1_tw", "delta_corrected_common_cr1_tw", "delta_legacy_covariance",
        "delta_pinned_source", "delta_region_cr1_context", "delta_independent_HC1_positive_control",
        "independent_scale_calibrated", "truth_trained_bias_centered_split_calibrated",
        "estimator_only_common_bias", "estimator_only_common_bias_bias_centered",
        "year_half_split_q_diagnostic"]
    summaries = []
    for method in methods:
        eval_rows = te[:year_n] if method == "year_half_split_q_diagnostic" else te
        summaries.append(summarize_panel(te, method, scname, seed, eval_rows=eval_rows))
    pairs = []
    reference = "delta_corrected_cr1_tw"
    for method in methods:
        eval_rows = te[:year_n] if method == "year_half_split_q_diagnostic" else te
        ca = np.array([r.get("cover__" + method, 0) for r in eval_rows], float)
        cb = np.array([r.get("cover__" + reference, 0) for r in eval_rows], float)
        dcover = ca - cb
        wa = np.array([r.get("width__" + method, np.nan) for r in eval_rows], float)
        wb = np.array([r.get("width__" + reference, np.nan) for r in eval_rows], float)
        width_ok = np.isfinite(wa) & np.isfinite(wb)
        wd = wa[width_ok] - wb[width_ok]
        pairs.append({"scenario": scname, "design_seed": seed, "method": method,
            "reference": reference, "n_planned": len(eval_rows),
            "coverage_diff": float(dcover.mean()) if len(dcover) else float("nan"),
            "coverage_diff_mcse_paired_1se": float(dcover.std(ddof=1) / np.sqrt(len(dcover))) if len(dcover) > 1 else float("nan"),
            "n_width_pairs": int(width_ok.sum()), "n_width_infinite_method": int(np.isinf(wa).sum()),
            "n_width_infinite_reference": int(np.isinf(wb).sum()),
            "mean_width_diff": float(wd.mean()) if len(wd) else float("nan"),
            "width_diff_mcse_paired_1se": float(wd.std(ddof=1) / np.sqrt(len(wd))) if len(wd) > 1 else float("nan")})
    allrows = tr + cal + te
    for r in allrows:
        err, h = r.get("error", float("nan")), r.get("h_corrected", float("nan"))
        good = r.get("fit_ok", False) and np.isfinite(err) and np.isfinite(h) and h > 0
        r["error_common_bias"] = err + shift if np.isfinite(err) else float("nan")
        r["A_hat_common_bias"] = r["A_hat"] + shift if np.isfinite(r["A_hat"]) else float("nan")
        r["score_raw"] = abs(err) / h if good else float("nan")
        r["score_bias_centered"] = abs(err - bias) / h if good and np.isfinite(bias) else float("nan")
        r["score_common_bias_raw"] = abs(err + shift) / h if good else float("nan")
        r["score_common_bias_centered"] = abs(err + shift - bias_common) / h if good and np.isfinite(bias_common) else float("nan")
        r["bias_hat_train"] = bias
        r["bias_hat_train_common_bias"] = bias_common
        r["q_raw_panel"] = qraw
        r["q_bias_centered_panel"] = qb
        r["q_raw_common_bias_panel"] = qraw_common
        r["q_bias_centered_common_bias_panel"] = qb_common
        r["mode_id"] = mode_id
    panel = {"mode": mode, "mode_id": mode_id, "scenario": scname, "scenario_id": int(sc["id"]),
        "design_seed": seed, "A_true": D["truth"], "design_file": D["design_path"],
        "design_sha256": D["design_sha256"], "mean_H": D["mean_H"], "frac_hot": D["frac_hot"],
        "positive_control_fit_ok": D["positive_control_fit_ok"],
        "positive_control_projection_rmse": D["source_projection_rmse"],
        "positive_control_beta_max_abs_error": D["positive_control_beta_max_abs_error"],
        "positive_control_mu_max_abs_error": D["positive_control_mu_max_abs_error"],
        "bias_hat_train": bias, "bias_hat_train_common_bias": bias_common,
        "n_bias_train_valid": len(train_errors), "n_scale_scores": len(raw_scores),
        "n_bias_scores": len(centered_scores), "q_raw": qraw, "q_bias_centered": qb,
        "q_raw_common_bias": qraw_common, "q_bias_centered_common_bias": qb_common,
        "q_bias_centered_common_bias_difference": qb_common - qb,
        "max_abs_centered_score_change_after_common_bias": max_centered_score_change,
        "max_abs_centered_estimator_change_after_common_bias": common_centered_maxdiff,
        "n_test": len(te), "yearhalf_test_reps": year_n,
        "yearhalf_ksplit": int(cfg["year_split_diagnostic"]["ksplit"]),
        "yearhalf_valid_splits_total": sum(r["yearhalf_splits_valid"] for r in te),
        "yearhalf_failed_splits_total": sum(r["yearhalf_splits_failed"] for r in te),
        "alpha": alpha, "estimator_only_common_bias": shift,
        "equivariance_max_error_delta": equiv_error,
        "source_sha256": sha(CORE_HD), "core_sim_sha256": sha(CORE_SIM)}
    panelfile = run_root() / "results" / f"heatday_{mode}_panel_{scname}_seed{seed}.csv"
    engine.write_csv(panelfile, allrows)
    return {"panel": panel, "summaries": summaries, "paired": pairs, "panelfile": str(panelfile)}


def preformal_manifest_path() -> Path:
    return ROOT / "results" / "heatday_preformal_manifest.json"


def pin_preformal() -> dict[str, Any]:
    cfg = read_config()
    if cfg["protocol_status"] != "approved":
        raise RuntimeError(f"heat-day preformal pin requires protocol_status='approved', got {cfg['protocol_status']!r}")
    out = preformal_manifest_path()
    if out.exists():
        raise RuntimeError(f"heat-day preformal manifest already exists: {out}")
    required = {
        "pilot_summary": ROOT / "results" / "heatday_pilot_summary.csv",
        "pilot_calibration": ROOT / "results" / "heatday_pilot_calibration.csv",
        "pilot_reproducibility": ROOT / "results" / "heatday_pilot_reproducibility.json",
        "preformal_review": ROOT / "docs" / "heatday_preformal_review.md",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"heat-day preformal artifacts missing: {missing}")
    payload = {"study_id": cfg["study_id"], "protocol_status": cfg["protocol_status"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha(CONFIG),
        "code_sha256": sha(Path(__file__).resolve()), "engine_sha256": sha(ENGINE_FILE),
        "core_hd_sha256": sha(CORE_HD), "core_sim_sha256": sha(CORE_SIM),
        "run_fail_hd_sha256": sha(RUN_FAIL_HD), "core_fail_hd_sha256": sha(CORE_FAIL_HD),
        "pilot_artifacts": {k: {"path": str(v), "sha256": sha(v)} for k, v in required.items()},
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
    write_json(out, payload)
    return payload


def run(mode: str, workers: int) -> None:
    cfg = read_config()
    manifest = preformal_manifest_path()
    if mode == "formal":
        if cfg["protocol_status"] != "approved":
            raise RuntimeError(f"heat-day formal run blocked by protocol_status={cfg['protocol_status']!r}")
        if not manifest.exists():
            raise RuntimeError("heat-day formal run blocked: no preformal manifest")
        pin = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {"config_sha256": sha(CONFIG), "engine_sha256": sha(ENGINE_FILE), "core_hd_sha256": sha(CORE_HD),
            "core_sim_sha256": sha(CORE_SIM), "run_fail_hd_sha256": sha(RUN_FAIL_HD),
            "core_fail_hd_sha256": sha(CORE_FAIL_HD)}
        diff = {k: (pin.get(k), v) for k, v in expected.items() if pin.get(k) != v}
        if diff:
            raise RuntimeError(f"heat-day preformal hash mismatch: {diff}")
    else:
        pin = None
    workers = max(1, min(int(workers), 4, int(cfg["execution"]["max_processes"])))
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    tasks = [(s["name"], int(seed), mode) for s in cfg["design"]["scenarios"]
        for seed in cfg["design"]["fixed_design_seeds"]]
    if workers == 1:
        results = [run_panel(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_panel, tasks))
    base = run_root() / "results"
    panels = [r["panel"] for r in results]
    summaries = [x for r in results for x in r["summaries"]]
    paired = [x for r in results for x in r["paired"]]
    engine.write_csv(base / f"heatday_{mode}_calibration.csv", panels)
    engine.write_csv(base / f"heatday_{mode}_summary.csv", summaries)
    engine.write_csv(base / f"heatday_{mode}_paired_differences.csv", paired)
    log = {"study_id": cfg["study_id"], "mode": mode, "mode_id": int(cfg["sampling"]["mode_ids"][mode]),
        "protocol_status": cfg["protocol_status"], "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - t0,
        "config_sha256": sha(CONFIG), "code_sha256": sha(Path(__file__).resolve()),
        "engine_sha256": sha(ENGINE_FILE), "core_hd_sha256": sha(CORE_HD), "core_sim_sha256": sha(CORE_SIM),
        "run_fail_hd_sha256": sha(RUN_FAIL_HD), "core_fail_hd_sha256": sha(CORE_FAIL_HD),
        "preformal_manifest_sha256": sha(manifest) if pin is not None else None,
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "workers": workers,
        "blas_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "panels": [r["panelfile"] for r in results],
        "warnings": ["Truth-trained bias/scale calibration uses simulation truth and is not deployable without a labeled calibration target.",
            "All scenarios use the same fixed H/Hcf exposure design for each scenario seed and set counterfactual cfg.bias=0. core_hd.common_bias changes eta0 for both factual and counterfactual means and can change scenario-specific A_true; A_true remains fixed only across outcome resamples within a design. The separate estimator-only shared_additive_shift diagnostic leaves A_true unchanged.",
            "Year-half q reuses test outcomes and is a descriptive diagnostic only."]}
    write_json(base / f"heatday_{mode}_reproducibility.json", log)
    print(json.dumps({"mode": mode, "elapsed_seconds": log["elapsed_seconds"], "panels": len(results)}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pin-preformal", action="store_true")
    args = ap.parse_args()
    try:
        if args.pin_preformal:
            payload = pin_preformal()
            print(json.dumps({k: payload[k] for k in ("config_sha256", "code_sha256", "engine_sha256", "core_hd_sha256")}, indent=2), flush=True)
        else:
            run(args.mode, args.workers)
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

