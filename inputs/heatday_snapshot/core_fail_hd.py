"""
core_fail_hd.py -- failure map on the heat-day x mortality DGP (independent family).

Same five interval constructions as core_fail.py, but the data are monthly heat days
(H) and a mortality-scale outcome, i.e. a different exposure construct and outcome
than the temperature x hospitalisation family.
"""
from __future__ import annotations

import numpy as np

from core_sim import Cfg, fit_glm, conformal_q, tq
from core_hd import simulate_hd, build_X, cf_X

NC = 17  # 1 + trend + 4 spline (2 lags x 2) + 11 month


def _reg_eff(fit, R):
    out = np.zeros(R)
    for r in range(1, R):
        out[r] = fit["beta"][NC + r - 1]
    return out


def _attrib(beta, re, Xc, Xcfc, off_m, R, T):
    eta = (Xc @ beta).reshape(R, T) + re[:, None] + off_m
    ecf = (Xcfc @ beta).reshape(R, T) + re[:, None] + off_m
    mf = np.exp(eta); mc = np.exp(ecf)
    return float((mf - mc).sum()), mf.reshape(-1), mc.reshape(-1)


def run_rep_fail_hd(cfg: Cfg, seed: int, B: int = 100, alpha: float = 0.10, ksplit: int = 20) -> dict:
    dat = simulate_hd(cfg, seed)
    rng = np.random.default_rng(seed + 991)
    C, R, T = cfg.n_c, cfg.R, cfg.T
    knots = np.quantile(dat["H"][:, :, :, 0].reshape(-1), [0.50, 0.85, 0.97])

    A_c = np.zeros(C); h_c = np.zeros(C)
    fits = []; metas = []
    for c in range(C):
        H = dat["H"][c].reshape(-1, 2); Hcf = dat["Hcf"][c].reshape(-1, 2)
        y = dat["y"][c].reshape(-1); off = np.log(np.repeat(dat["pop"][c], T))
        region = np.repeat(np.arange(R), T)
        month = np.tile(np.arange(T) % 12, R); trend = np.tile(np.arange(T) / 12.0, R)
        X = build_X(H, region, month, trend, knots, R)
        fit = fit_glm(y, X, off, region, cluster2=np.tile(np.arange(T) // 12, R))
        Xcf = cf_X(X, Hcf, knots)
        mu = np.exp(np.clip(X @ fit["beta"] + off, -30, 30))
        mucf = np.exp(np.clip(Xcf @ fit["beta"] + off, -30, 30))
        A_c[c] = mu.sum() - mucf.sum()
        g = X.T @ mu - Xcf.T @ mucf
        h_c[c] = float(np.sqrt(max(g @ fit["V"] @ g, 0.0)))
        fits.append(fit); metas.append(dict(X=X, Xcf=Xcf, off=off, region=region, y=y))

    A_hat = float(A_c.sum()); h_tot = float(np.sqrt((h_c**2).sum()))
    G = min(f["G"] for f in fits)
    qt = tq(max(G - 1, 3))

    boots = np.zeros(B)
    for c in range(C):
        fit = fits[c]; meta = metas[c]
        X = meta["X"]; off = meta["off"]
        e = (meta["y"] - fit["mu"]).reshape(R, T)
        for b in range(B):
            perm = rng.integers(0, R, R)
            beta_b = fit["beta"] + fit["Ainv"] @ (X.T @ e[perm].reshape(-1))
            boots[b] += float((np.exp(np.clip(X @ beta_b + off, -30, 30))
                               - np.exp(np.clip(meta["Xcf"] @ beta_b + off, -30, 30))).sum())
    q_boot = float(np.quantile(np.abs(boots - A_hat) / h_tot, 1 - alpha))

    years = np.tile(np.arange(T) // 12, R)
    z_year = []
    for k in range(ksplit):
        keep = rng.permutation(10)[:5]
        m1 = np.isin(years, keep); m2 = ~m1
        A1 = A2 = v1 = v2 = 0.0
        for c in range(C):
            fit = fits[c]; meta = metas[c]
            X = meta["X"]; off = meta["off"]; reg = meta["region"]; y = meta["y"]
            f1 = fit_glm(y[m1], X[m1], off[m1], reg[m1], beta0=fit["beta"], iters=12)
            f2 = fit_glm(y[m2], X[m2], off[m2], reg[m2], beta0=fit["beta"], iters=12)
            re = _reg_eff(fit, R); off_m = off.reshape(R, T)
            Xc = X[:, :NC]; Xcfc = meta["Xcf"][:, :NC]
            a1, mf1, mc1 = _attrib(f1["beta"][:NC], re, Xc, Xcfc, off_m, R, T)
            a2, mf2, mc2 = _attrib(f2["beta"][:NC], re, Xc, Xcfc, off_m, R, T)
            g1 = Xc.T @ mf1 - Xcfc.T @ mc1; g2 = Xc.T @ mf2 - Xcfc.T @ mc2
            v1 += float(g1 @ f1["V"][:NC, :NC] @ g1); v2 += float(g2 @ f2["V"][:NC, :NC] @ g2)
            A1 += a1; A2 += a2
        if v1 + v2 > 0:
            z_year.append(abs(A1 - A2) / np.sqrt(v1 + v2))
    q_year = conformal_q(np.asarray(z_year), alpha)

    z_state = []
    pen = np.eye(NC + R - 1) * 1e-6
    for k in range(ksplit):
        perm = rng.permutation(R); half = set(perm[: R // 2].tolist())
        A1 = A2 = v1 = v2 = 0.0
        for c in range(C):
            fit = fits[c]; meta = metas[c]
            X = meta["X"]; off = meta["off"]; reg = meta["region"]; y = meta["y"]
            m1 = np.isin(reg, list(half)); m2 = ~m1
            if m1.sum() < 50 or m2.sum() < 50:
                continue
            f1 = fit_glm(y[m1], X[m1], off[m1], reg[m1], beta0=fit["beta"], pen=pen, iters=10)
            f2 = fit_glm(y[m2], X[m2], off[m2], reg[m2], beta0=fit["beta"], pen=pen, iters=10)
            re = _reg_eff(fit, R); off_m = off.reshape(R, T)
            Xc = X[:, :NC]; Xcfc = meta["Xcf"][:, :NC]
            a1, mf1, mc1 = _attrib(f1["beta"][:NC], re, Xc, Xcfc, off_m, R, T)
            a2, mf2, mc2 = _attrib(f2["beta"][:NC], re, Xc, Xcfc, off_m, R, T)
            g1 = Xc.T @ mf1 - Xcfc.T @ mc1; g2 = Xc.T @ mf2 - Xcfc.T @ mc2
            v1 += float(g1 @ f1["V"][:NC, :NC] @ g1); v2 += float(g2 @ f2["V"][:NC, :NC] @ g2)
            A1 += a1; A2 += a2
        if v1 + v2 > 0:
            z_state.append(abs(A1 - A2) / np.sqrt(v1 + v2))
    q_state = conformal_q(np.asarray(z_state), alpha) if len(z_state) >= 5 else np.nan

    q_two = max(q_year, q_state) if np.isfinite(q_state) else q_year
    out = dict(A_true=dat["A_true"], A_hat=A_hat, h=h_tot, q_year=q_year, q_state=q_state,
               q_two=q_two, q_boot=q_boot)
    out["cc_year"] = (A_hat - q_year * h_tot, A_hat + q_year * h_tot)
    out["cc_state"] = (A_hat - q_state * h_tot, A_hat + q_state * h_tot) if np.isfinite(q_state) else (np.nan, np.nan)
    out["cc_two"] = (A_hat - q_two * h_tot, A_hat + q_two * h_tot)
    out["boot_basic"] = (A_hat - q_boot * h_tot, A_hat + q_boot * h_tot)
    out["fixed13"] = (A_hat - 1.3 * qt * h_tot, A_hat + 1.3 * qt * h_tot)
    out["delta"] = (A_hat - qt * h_tot, A_hat + qt * h_tot)
    return out
