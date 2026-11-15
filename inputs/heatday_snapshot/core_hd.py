"""
core_hd.py -- heat-day exposure version of the conformal attribution simulation (V2 pre-registration).

Exposure: H = number of days in the month with daily Tmax above the region's factual-period P95.
Counterfactual: H_cf = same days after subtracting the anthropogenic increment Delta.
Truth: A_true = sum(mu(H) - mu(H_cf)) with the true parameters.

Estimator: quasi-Poisson GLM, region FE + month FE + trend + natural spline on H (lag 0-1, 2 df each),
two-way (region x year) CGM covariance with PSD projection.

Intervals: Delta / Bootstrap / Bayesian / C-split / C-block / C-weighted / CC (country-level calibration).
"""
from __future__ import annotations

import numpy as np

from core_sim import Cfg, fit_glm, conformal_q, tq, ns_basis, _ar1


# ---------------------------------------------------------------- DGP
def simulate_hd(cfg: Cfg, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    C, R, T = cfg.n_c, cfg.R, cfg.T
    Rn = C * R
    tgrid = np.arange(T)
    months = tgrid % 12

    base = rng.uniform(15.0, 26.0, C)
    amp = rng.uniform(3.0, 8.0, C)
    phase = rng.uniform(0, 2 * np.pi, C)
    rate = rng.uniform(cfg.rate_lo, cfg.rate_hi, C)
    d0 = rng.uniform(0.8, 1.6, C) * cfg.delta_mult
    d1 = rng.uniform(0.2, 0.5, C) * cfg.delta_mult
    gcm = rng.normal(0.0, cfg.sd_gcm, C)

    reg_c = np.repeat(np.arange(C), R)
    alpha = np.log(rate[reg_c]) + rng.normal(0, cfg.sd_alpha, Rn)
    reg_amp = amp[reg_c] * np.exp(rng.normal(0, 0.25, Rn))
    pop = np.exp(rng.normal(np.log(2.5e6), 0.6, Rn))
    slope = rng.normal(0, cfg.sd_slope, Rn)
    sd_day = 3.0 * np.exp(rng.normal(0, 0.15, Rn))

    # monthly mean temperature (t = -1 .. T-1, one pre-sample month for lag 1)
    TT = T + 1
    tt = np.arange(-1, T)
    seasonal = base[reg_c, None] + reg_amp[:, None] * np.cos(
        2 * np.pi * (tt[None, :] % 12 - phase[reg_c, None]) / 12.0
    )
    warm = rng.uniform(0.02, 0.06, C)[reg_c, None] * (tt[None, :] / 12.0)
    common = _ar1(rng, TT, 0.6, cfg.sd_common, C)[reg_c]
    local = _ar1(rng, TT, cfg.phi_ar, cfg.sd_ar, Rn)
    m = seasonal + warm + common + local + rng.normal(0, 0.10, (Rn, TT))

    # daily temperatures: (Rn, TT, 30)
    daily = rng.normal(m[:, :, None], sd_day[:, None, None], size=(Rn, TT, 30))
    thr = np.quantile(daily.reshape(Rn, -1), 0.95, axis=1)         # per-region factual P95
    H_all = (daily > thr[:, None, None]).sum(axis=2).astype(float)

    delta_t = d0[reg_c, None] + d1[reg_c, None] * (tt[None, :] / T)
    delta_true = delta_t + gcm[reg_c, None]
    delta_hat = delta_true + cfg.bias
    Hcf_all = (daily - delta_hat[:, :, None] > thr[:, None, None]).sum(axis=2).astype(float)

    # outcome: g(H) = 0.020 H + 0.0010 H^2, lags (0.6, 0.4)
    def g(h):
        return 0.020 * h + 0.0010 * h * h

    w = np.array([0.6, 0.4])
    gf = w[0] * g(H_all[:, 1:]) + w[1] * g(H_all[:, :-1])
    gc = w[0] * g(Hcf_all[:, 1:]) + w[1] * g(Hcf_all[:, :-1])

    eta0 = alpha[:, None] + 0.10 * np.cos(2 * np.pi * (months[None, :] - phase[reg_c, None]) / 12.0)
    eta0 = eta0 + slope[:, None] * tgrid[None, :]
    if cfg.misspec:
        zx = (H_all[:, 1:] - H_all[:, 1:].mean()) / (H_all[:, 1:].std() + 1e-9)
        U = -0.6 * zx + rng.normal(0, 0.5, (Rn, T))
        eta0 = eta0 + 0.18 * U
    if cfg.common_bias:
        hm = H_all[:, 1:].reshape(C, R, T).mean(axis=1).copy()
        mon = np.arange(T) % 12
        for cc in range(C):
            for mm in range(12):
                idx = mon == mm
                hm[cc, idx] = hm[cc, idx] - hm[cc, idx].mean()
        zc = hm / (hm.std(axis=1, keepdims=True) + 1e-9)
        eta0 = eta0 + cfg.common_bias * zc[reg_c]

    mu = pop[:, None] * np.exp(eta0 + gf)
    mucf = pop[:, None] * np.exp(eta0 + gc)
    y = rng.negative_binomial(cfg.theta, cfg.theta / (cfg.theta + mu))

    H0 = H_all[:, 1:]
    H1 = H_all[:, :-1]
    Hc0 = Hcf_all[:, 1:]
    Hc1 = Hcf_all[:, :-1]
    A_true_c = (mu - mucf).reshape(C, R, T).sum(axis=(1, 2))

    return dict(
        y=y.reshape(C, R, T), mu=mu.reshape(C, R, T), mucf=mucf.reshape(C, R, T),
        H=np.stack([H0, H1], axis=-1).reshape(C, R, T, 2),
        Hcf=np.stack([Hc0, Hc1], axis=-1).reshape(C, R, T, 2),
        pop=pop.reshape(C, R), thr=thr.reshape(C, R),
        A_true_c=A_true_c, A_true=float(A_true_c.sum()),
        mean_H=float(H0.mean()), frac_hot=float((H0 > 0).mean()),
    )


# ---------------------------------------------------------------- design
def build_X(H, region, month, trend, knots, R):
    """columns: [1, trend, 2 spline lag0, 2 spline lag1, 11 month, R-1 region] -> common = 17."""
    n = len(month)
    cols = [np.ones(n), trend]
    for l in range(2):
        cols.append(ns_basis(H[:, l], knots)[:, 1:])
    for mm in range(1, 12):
        cols.append((month == mm).astype(float))
    for r in range(1, R):
        cols.append((region == r).astype(float))
    return np.column_stack(cols)


def cf_X(X, Hcf, knots, shift=0.0):
    Xc = X.copy()
    for l in range(2):
        Xc[:, 2 + 2 * l : 2 + 2 * (l + 1)] = ns_basis(Hcf[:, l] - shift, knots)[:, 1:]
    return Xc


# ---------------------------------------------------------------- one replication
def run_rep_hd(cfg: Cfg, seed: int, B: int = 150, alpha: float = 0.10, ksplit: int = 30) -> dict:
    dat = simulate_hd(cfg, seed)
    rng = np.random.default_rng(seed + 991)
    C, R, T = cfg.n_c, cfg.R, cfg.T
    knots = np.quantile(dat["H"][:, :, :, 0].reshape(-1), [0.50, 0.85, 0.97])

    A_c = np.zeros(C); h_c = np.zeros(C)
    fit_list = []; meta_list = []
    for c in range(C):
        H = dat["H"][c].reshape(-1, 2)
        Hcf = dat["Hcf"][c].reshape(-1, 2)
        y = dat["y"][c].reshape(-1)
        off = np.log(np.repeat(dat["pop"][c], T))
        region = np.repeat(np.arange(R), T)
        month = np.tile(np.arange(T) % 12, R)
        trend = np.tile(np.arange(T) / 12.0, R)
        X = build_X(H, region, month, trend, knots, R)
        fit = fit_glm(y, X, off, region, cluster2=np.tile(np.arange(T) // 12, R))
        Xcf = cf_X(X, Hcf, knots)
        mu = np.exp(np.clip(X @ fit["beta"] + off, -30, 30))
        mucf = np.exp(np.clip(Xcf @ fit["beta"] + off, -30, 30))
        A_c[c] = mu.sum() - mucf.sum()
        g = X.T @ mu - Xcf.T @ mucf
        h_c[c] = float(np.sqrt(max(g @ fit["V"] @ g, 0.0)))
        fit_list.append(fit)
        meta_list.append(dict(X=X, Xcf=Xcf, off=off, region=region, y=y))

    A_hat = float(A_c.sum())
    h_tot = float(np.sqrt((h_c**2).sum()))
    G = min(f["G"] for f in fit_list)
    qt = tq(max(G - 1, 3))
    out = dict(cfg=cfg.label, seed=seed, A_true=dat["A_true"], A_true_c=dat["A_true_c"],
               A_hat=A_hat, h_tot=h_tot, h_c=h_c, A_c=A_c,
               mean_H=dat["mean_H"], frac_hot=dat["frac_hot"])

    out["D"] = (A_hat - qt * h_tot, A_hat + qt * h_tot)
    out["D_c"] = np.column_stack([A_c - qt * h_c, A_c + qt * h_c])

    # ---------------- bootstrap ----------------
    boots = np.zeros((B, C))
    for c in range(C):
        fit = fit_list[c]; meta = meta_list[c]
        X = meta["X"]; off = meta["off"]; reg = meta["region"]
        e = (meta["y"] - fit["mu"]).reshape(R, T)
        for b in range(B):
            perm = rng.integers(0, R, R)
            beta_b = fit["beta"] + fit["Ainv"] @ (X.T @ e[perm].reshape(-1))
            mub = np.exp(np.clip(X @ beta_b + off, -30, 30))
            mucfb = np.exp(np.clip(meta["Xcf"] @ beta_b + off, -30, 30))
            boots[b, c] = mub.sum() - mucfb.sum()
    btot = boots.sum(1)
    out["B"] = (float(np.quantile(btot, alpha / 2)), float(np.quantile(btot, 1 - alpha / 2)))
    out["B_c"] = np.column_stack([np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)])
    out["boot_sd"] = float(np.std(btot))

    # ---------------- Bayesian (Laplace + RW) ----------------
    D2 = np.array([[1.0, -1.0]])
    penb = 0.5 * (D2.T @ D2)
    draws = np.zeros((1000, C))
    for c in range(C):
        fit = fit_list[c]; meta = meta_list[c]
        X = meta["X"]; off = meta["off"]
        p = X.shape[1]
        pen = np.eye(p) * 1e-4
        for l in range(2):
            s = 2 + 2 * l
            pen[s : s + 2, s : s + 2] += penb
        fb = fit_glm(meta["y"], X, off, meta["region"], beta0=fit["beta"], pen=pen, iters=15)
        wgt = np.maximum(fb["mu"], 1e-8)
        Sigma = fit["phi"] * np.linalg.inv(X.T @ (wgt[:, None] * X) + pen)
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
        bd = fb["beta"][None, :] + rng.normal(size=(1000, p)) @ L.T
        muf = np.exp(np.clip(bd @ X.T + off, -30, 30))
        mucf = np.exp(np.clip(bd @ meta["Xcf"].T + off, -30, 30))
        draws[:, c] = muf.sum(1) - mucf.sum(1)
    dt = draws.sum(1)
    out["Y"] = (float(np.quantile(dt, alpha / 2)), float(np.quantile(dt, 1 - alpha / 2)))
    out["Y_c"] = np.column_stack([np.quantile(draws, alpha / 2, axis=0), np.quantile(draws, 1 - alpha / 2, axis=0)])

    # ---------------- conformal ----------------
    z_units = []
    for c in range(C):
        fit = fit_list[c]; meta = meta_list[c]
        mu_all = fit["mu"].reshape(R, T)
        y = meta["y"].reshape(R, T)
        for r in range(R):
            z_units.extend((np.abs(y[r] - mu_all[r]) / np.sqrt(np.maximum(fit["phi"] * mu_all[r], 1e-8))).tolist())
    q_split = conformal_q(np.asarray(z_units), alpha)

    H3 = dat["H"][:, :, :, 0].reshape(C, R, T)
    H3c = dat["Hcf"][:, :, :, 0].reshape(C, R, T)
    mh = float(H3.mean()); mhc = float(H3c.mean()); sh = float(H3.std()) + 1e-9
    Wimp = np.exp(-0.5 * ((H3 - mhc) / sh) ** 2 + 0.5 * ((H3 - mh) / sh) ** 2)
    Wimp = np.clip(Wimp / Wimp.mean(), 0.5, 2.0)
    ess = float(Wimp.sum() ** 2 / (Wimp**2).sum())

    years = np.tile(np.arange(T) // 12, R)
    z_blk = []; z_w = []; z_ctry = []
    for k in range(ksplit):
        keep = rng.permutation(10)[:5]
        m1 = np.isin(years, keep)
        m2 = ~m1
        A1 = A2 = v1 = v2 = 0.0
        Aw1 = Aw2 = vw1 = vw2 = 0.0
        for c in range(C):
            fit = fit_list[c]; meta = meta_list[c]
            X = meta["X"]; off = meta["off"]; reg = meta["region"]; y = meta["y"]
            f1 = fit_glm(y[m1], X[m1], off[m1], reg[m1], beta0=fit["beta"], iters=12,
                         cluster2=np.tile(np.arange(T) // 12, R)[m1])
            f2 = fit_glm(y[m2], X[m2], off[m2], reg[m2], beta0=fit["beta"], iters=12,
                         cluster2=np.tile(np.arange(T) // 12, R)[m2])
            reg_eff = np.zeros(R)
            for r in range(1, R):
                reg_eff[r] = fit["beta"][17 + r - 1]
            off_m = off.reshape(R, T)
            Xc = X[:, :17]; Xcfc = meta["Xcf"][:, :17]
            wc = Wimp[c].reshape(-1)
            a1c = v1c = a2c = v2c = 0.0
            for f, tag in ((f1, 1), (f2, 2)):
                eta = (Xc @ f["beta"][:17]).reshape(R, T) + reg_eff[:, None] + off_m
                ecf = (Xcfc @ f["beta"][:17]).reshape(R, T) + reg_eff[:, None] + off_m
                mf = np.exp(eta); mc = np.exp(ecf)
                d = (mf - mc).reshape(-1)
                A = float(d.sum())
                g = Xc.T @ mf.reshape(-1) - Xcfc.T @ mc.reshape(-1)
                v = float(g @ f["V"][:17, :17] @ g)
                Aw = float((wc * d).sum())
                gw = Xc.T @ (wc * mf.reshape(-1)) - Xcfc.T @ (wc * mc.reshape(-1))
                vw = float(gw @ f["V"][:17, :17] @ gw)
                if tag == 1:
                    A1 += A; v1 += v; Aw1 += Aw; vw1 += vw; a1c += A; v1c += v
                else:
                    A2 += A; v2 += v; Aw2 += Aw; vw2 += vw; a2c += A; v2c += v
            z_ctry.append(abs(a1c - a2c) / np.sqrt(max(v1c + v2c, 1e-12)))
        z_blk.append(abs(A1 - A2) / np.sqrt(max(v1 + v2, 1e-12)))
        z_w.append(abs(Aw1 - Aw2) / np.sqrt(max(vw1 + vw2, 1e-12)))

    q_block = conformal_q(np.asarray(z_blk), alpha)
    q_w = conformal_q(np.asarray(z_w), alpha)
    q_ctry = conformal_q(np.asarray(z_ctry), alpha)

    out["CS"] = (A_hat - q_split * h_tot, A_hat + q_split * h_tot)
    out["CB"] = (A_hat - q_block * h_tot, A_hat + q_block * h_tot)
    out["CW"] = (A_hat - q_w * h_tot, A_hat + q_w * h_tot)
    out["CC"] = (A_hat - q_ctry * h_tot, A_hat + q_ctry * h_tot)
    out["CS_c"] = np.column_stack([A_c - q_split * h_c, A_c + q_split * h_c])
    out["CB_c"] = np.column_stack([A_c - q_block * h_c, A_c + q_block * h_c])
    out["CW_c"] = np.column_stack([A_c - q_w * h_c, A_c + q_w * h_c])
    out["CC_c"] = np.column_stack([A_c - q_ctry * h_c, A_c + q_ctry * h_c])
    out["q"] = np.array([q_split, q_block, q_w, q_ctry])
    out["ess"] = ess
    return out
