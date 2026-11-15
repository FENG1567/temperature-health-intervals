"""run_fail_hd.py -- failure map on the heat-day x mortality family."""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_sim import Cfg
from core_fail_hd import run_rep_fail_hd

METHODS = ["delta", "cc_year", "cc_state", "cc_two", "boot_basic", "fixed13"]
BASE = dict(rate_lo=3e-6, rate_hi=8e-6, theta=20.0)
SCEN = {
    "G0": dict(**BASE),
    "G1": dict(**BASE, phi_ar=0.9, sd_ar=1.0),
    "G2": dict(**BASE, sd_common=1.0),
    "G3": dict(**BASE, sd_alpha=0.8),
    "G4": dict(**BASE, common_bias=0.25),
    "G5": dict(**BASE, phi_ar=0.9, sd_ar=1.0, sd_common=1.0, sd_alpha=0.8, common_bias=0.25),
    "G6": dict(**BASE, common_bias=0.06),
    "G7": dict(**BASE, common_bias=0.12),
}
IDX = {"G0": 61, "G1": 62, "G2": 63, "G3": 64, "G4": 65, "G5": 66, "G6": 67, "G7": 68}


def one(args):
    label, R, seed = args
    cfg = Cfg(R=R, label=label, **SCEN[label])
    r = run_rep_fail_hd(cfg, seed, B=100, ksplit=20)
    out = {"A_true": r["A_true"], "A_hat": r["A_hat"], "h": r["h"],
           "q_year": r["q_year"], "q_state": r["q_state"], "q_two": r["q_two"], "q_boot": r["q_boot"]}
    for m in METHODS:
        out[f"lo_{m}"], out[f"hi_{m}"] = r[m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--outdir", default="/data/admin/fengshuo/conformal_attribution_20261107/simulation/results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"FAILHD_{a.label}_R{a.R}.npz")
    if os.path.exists(path):
        print("exists", path); return
    seeds = [20500101 + 100000 * IDX[a.label] + 1000 * a.R + i for i in range(a.reps)]
    t0 = time.time(); res = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(a.label, a.R, s) for s in seeds], chunksize=5)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {a.label}: {i+1}/{a.reps} ({time.time()-t0:.0f}s)", flush=True)
    arr = {k: np.stack([r[k] for r in res]) for k in res[0]}
    np.savez_compressed(path, **arr)
    summ = dict(label=a.label, R=a.R, reps=a.reps, seconds=round(time.time() - t0, 1),
                A_true_mean=float(arr["A_true"].mean()), bias=float((arr["A_hat"] - arr["A_true"]).mean()),
                rmse=float(np.sqrt(((arr["A_hat"] - arr["A_true"]) ** 2).mean())))
    for m in METHODS:
        cov = float(np.mean((arr[f"lo_{m}"] <= arr["A_true"]) & (arr["A_true"] <= arr[f"hi_{m}"])))
        wid = float(np.mean(arr[f"hi_{m}"] - arr[f"lo_{m}"]))
        summ[f"cov_{m}"] = cov; summ[f"width_{m}"] = wid
    wb = summ["width_boot_basic"]
    for m in METHODS:
        summ[f"wratio_{m}"] = summ[f"width_{m}"] / wb if wb else None
    summ["q_year_mean"] = float(np.nanmean(arr["q_year"]))
    summ["q_state_mean"] = float(np.nanmean(arr["q_state"]))
    with open(os.path.join(a.outdir, f"FAILHD_{a.label}_R{a.R}.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()