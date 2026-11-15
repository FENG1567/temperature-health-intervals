"""
core_sim.py -- simulation core for conformal attribution intervals (v4).

DGP: 5 countries x regions x months, nonlinear + lagged exposure-response, known truth, NB2.
Estimation: quasi-Poisson GLM (region FE + month FE + trend + natural spline on lag0-2), IRLS + CR1.
Intervals: Delta / Bootstrap / Bayesian (Laplace + RW2) / Conformal (split, block, weighted).

Counterfactual x_cf = x - Delta_hat enters prediction only, never the fit.
Truth A_true is computed from the true parameters and the true Delta.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class Cfg:
    n_c: int = 5
    R: int = 16
    T: int = 120
    phi_ar: float = 0.5
    sd_ar: float = 0.8
    sd_common: float = 0.3
    theta: float = 40.0
    sd_gcm: float = 0.05
    bias: float = 0.0
    delta_mult: float = 1.0
    misspec: bool = False
    ext_frac: float = 0.5
    sd_slope: float = 0.002
    sd_alpha: float = 0.25
    common_bias: float = 0.0
    season_drift: float = 0.0
    delta_spatial: float = 0.0
    delta_seasonal: float = 0.0
    amp_lo: float = 3.0
    amp_hi: float = 8.0
    rate_lo: float = 6e-5
    rate_hi: float = 1.2e-4
    label: str = "S1"


# ---------------------------------------------------------------- spline / AR1
def ns_basis(x, knots):
    """Natural cubic spline (truncated power). Returns K columns: [1, x, ...]."""
    x = np.asarray(x, dtype=float)
    K = len(knots)
    cols = [np.ones_like(x), x]
    if K >= 3:
        d_last = (np.maximum(x - knots[-2], 0.0) ** 3 - np.maximum(x - knots[-1], 0.0) ** 3) / (
            knots[-1] - knots[-2]
        )
        for k in range(K - 2):
            dk = (np.maximum(x - knots[k], 0.0) ** 3 - np.maximum(x - knots[-1], 0.0) ** 3) / (
                knots[-1] - knots[k]
            )
            cols.append(dk - d_last)
    return np.column_stack(cols)


def _ar1(rng, n, phi, sd, n_region):
    e = rng.normal(0, sd, size=(n_region, n))
    out = np.empty_like(e)
    out[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        out[:, t] = phi * out[:, t - 1] + e[:, t]
    return out


def true_g(x, mmt, cold, heat, ext):
    return (
        cold * np.maximum(0.0, mmt - x)
        + heat * np.maximum(0.0, x - mmt)
        + ext * np.maximum(0.0, x - mmt - 8.0)
    )


# ---------------------------------------------------------------- DGP
def simulate(cfg: Cfg, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    C, R, T = cfg.n_c, cfg.R, cfg.T
    Rn = C * R
    tgrid = np.arange(T)
    months = tgrid % 12

    base = rng.uniform(15.0, 26.0, C)
    amp = rng.uniform(3.0, 8.0, C)
    phase = rng.uniform(0, 2 * np.pi, C)
    mmt = base + 0.10 * amp
    heat = rng.uniform(0.012, 0.030, C)
    cold = rng.uniform(0.004, 0.012, C)
    ext = heat * cfg.ext_frac
    rate = rng.uniform(cfg.rate_lo, cfg.rate_hi, C)
    d0 = rng.uniform(0.8, 1.6, C) * cfg.delta_mult
    d1 = rng.uniform(0.2, 0.5, C) * cfg.delta_mult
    gcm = rng.normal(0.0, cfg.sd_gcm, C)

    reg_c = np.repeat(np.arange(C), R)
    alpha = np.log(rate[reg_c]) + rng.normal(0, cfg.sd_alpha, Rn)
    reg_amp = amp[reg_c] * np.exp(rng.normal(0, 0.25, Rn))
    pop = np.exp(rng.normal(np.log(2.5e6), 0.6, Rn))
    slope = rng.normal(0, cfg.sd_slope, Rn)

    TT = T + 2
    tt = np.arange(-2, T)
    amp_t = reg_amp[:, None] * (1.0 + cfg.season_drift * (tt[None, :] / T - 0.5))
    seasonal = base[reg_c, None] + amp_t * np.cos(
        2 * np.pi * (tt[None, :] % 12 - phase[reg_c, None]) / 12.0
    )
    warm = rng.uniform(0.02, 0.06, C)[reg_c, None] * (tt[None, :] / 12.0)
    common = _ar1(rng, TT, 0.6, cfg.sd_common, C)[reg_c]
    local = _ar1(rng, TT, cfg.phi_ar, cfg.sd_ar, Rn)
    x_all = seasonal + warm + common + local + rng.normal(0, 0.10, (Rn, TT))
    delta_t = d0[reg_c, None] + d1[reg_c, None] * (tt[None, :] / T)
    z_r = rng.normal(0, 1, Rn)
    z_m = rng.normal(0, 1, 12)
    structure = 1.0 + cfg.delta_spatial * z_r[:, None] + cfg.delta_seasonal * z_m[tt % 12][None, :]
    delta_true = delta_t * structure + gcm[reg_c, None]
    delta_hat = delta_t + gcm[reg_c, None] + cfg.bias
    xcf_all = x_all - delta_true
    xcf_hat_all = x_all - delta_hat

    eta0 = alpha[:, None] + 0.10 * np.cos(2 * np.pi * (months[None, :] - phase[reg_c, None]) / 12.0)
    eta0 = eta0 + slope[:, None] * tgrid[None, :]
    lagw = np.array([0.5, 0.3, 0.2])
    gf = np.zeros((Rn, T))
    gc = np.zeros((Rn, T))
    for l in range(3):
        sl = slice(2 - l, 2 - l + T)
        gf += lagw[l] * true_g(x_all[:, sl], mmt[reg_c, None], cold[reg_c, None], heat[reg_c, None], ext[reg_c, None])
        gc += lagw[l] * true_g(xcf_all[:, sl], mmt[reg_c, None], cold[reg_c, None], heat[reg_c, None], ext[reg_c, None])

    if cfg.misspec:
        zx = (x_all[:, 2:] - x_all[:, 2:].mean()) / x_all[:, 2:].std()
        U = -0.6 * zx + rng.normal(0, 0.5, (Rn, T))
        eta0 = eta0 + 0.18 * U
    if cfg.common_bias:
        xm = x_all[:, 2:].reshape(C, R, T).mean(axis=1)
        zc = (xm - xm.mean(axis=1, keepdims=True)) / (xm.std(axis=1, keepdims=True) + 1e-9)
        eta0 = eta0 + cfg.common_bias * zc[reg_c]

    mu = pop[:, None] * np.exp(eta0 + gf)
    mucf = pop[:, None] * np.exp(eta0 + gc)
    y = rng.negative_binomial(cfg.theta, cfg.theta / (cfg.theta + mu))

    xl = np.stack([x_all[:, 2 - l : 2 - l + T] for l in range(3)], axis=-1)
    xcl = np.stack([xcf_all[:, 2 - l : 2 - l + T] for l in range(3)], axis=-1)
    xcl_hat = np.stack([xcf_hat_all[:, 2 - l : 2 - l + T] for l in range(3)], axis=-1)
    A_true_c = (mu - mucf).reshape(C, R, T).sum(axis=(1, 2))

    return dict(
        y=y.reshape(C, R, T), mu=mu.reshape(C, R, T), mucf=mucf.reshape(C, R, T),
        x_lag=xl.reshape(C, R, T, 3), xcf_lag=xcl.reshape(C, R, T, 3), xcf_hat_lag=xcl_hat.reshape(C, R, T, 3),
        pop=pop.reshape(C, R),
        delta_true=delta_true[:, 2:].reshape(C, R, T)[:, 0, :],
        delta_hat=delta_hat[:, 2:].reshape(C, R, T)[:, 0, :],
        A_true_c=A_true_c, A_true=float(A_true_c.sum()),
    )


# ---------------------------------------------------------------- design / fit
def build_X(xlag, region, month, trend, knots, n_region):
    """Column order: [1, trend, 9 spline (3 lags x 3), 11 month dummies, R-1 region dummies]."""
    n = len(month)
    cols = [np.ones(n), trend]
    for l in range(3):
        cols.append(ns_basis(xlag[:, l], knots)[:, 1:])
    for m in range(1, 12):
        cols.append((month == m).astype(float))
    for r in range(1, n_region):
        cols.append((region == r).astype(float))
    return np.column_stack(cols)


def cf_X(X, xcflag, knots, shift):
    """Replace the 9 spline columns of X by basis(xcf - shift)."""
    Xc = X.copy()
    for l in range(3):
        Xc[:, 2 + 2 * l : 2 + 2 * (l + 1)] = ns_basis(xcflag[:, l] - shift, knots)[:, 1:]
    return Xc


def fit_glm(y, X, offset, region, beta0=None, pen=None, iters=30, cluster2=None):
    n, p = X.shape
    b = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    for _ in range(iters):
        eta = np.clip(X @ b + offset, -30, 30)
        mu = np.exp(eta)
        w = np.maximum(mu, 1e-8)
        z = eta - offset + (y - mu) / w
        A = X.T @ (w[:, None] * X)
        if pen is not None:
            A = A + pen
        rhs = X.T @ (w * z)
        try:
            b_new = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            b_new = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    eta = np.clip(X @ b + offset, -30, 30)
    mu = np.exp(eta)
    w = np.maximum(mu, 1e-8)
    A = X.T @ (w[:, None] * X)
    if pen is not None:
        A = A + pen
    Ainv = np.linalg.inv(A)
    u = y - mu
    def _meat(idx):
        M = np.zeros((p, p))
        for g in np.unique(idx):
            m = idx == g
            sg = X[m].T @ u[m]
            M += np.outer(sg, sg)
        return M
    meat = _meat(region)
    if cluster2 is not None:
        XU = X * u[:, None]
        meat = meat + _meat(cluster2) - XU.T @ XU
        G = min(len(np.unique(region)), len(np.unique(cluster2)))
    else:
        G = len(np.unique(region))
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    V = Ainv @ meat @ Ainv * corr
    V = 0.5 * (V + V.T)
    ev, evec = np.linalg.eigh(V)
    V = (evec * np.maximum(ev, 0.0)) @ evec.T
    phi = float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-8)) / max(n - p, 1))
    return dict(beta=b, V=V, mu=mu, phi=max(phi, 1.0), p=p, n=n, G=G, Ainv=Ainv)


def tq(df):
    z = 1.6448536269514722
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def conformal_q(scores, alpha=0.10, weights=None):
    s = np.asarray(scores, float)
    n = len(s)
    if n == 0:
        return np.nan
    if weights is None:
        k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
        return float(np.sort(s)[k - 1])
    w = np.asarray(weights, float)
    w = w / w.sum()
    order = np.argsort(s)
    cw = np.cumsum(w[order])
    idx = min(int(np.searchsorted(cw, 1 - alpha)), n - 1)
    return float(s[order][idx])


# ---------------------------------------------------------------- one replication
def run_rep(cfg: Cfg, seed: int, B: int = 120, alpha: float = 0.10, boot: str = "score", ksplit: int = 30) -> dict:
    dat = simulate(cfg, seed)
    rng = np.random.default_rng(seed + 991)
    C, R, T = cfg.n_c, cfg.R, cfg.T

    knots = np.quantile(dat["x_lag"][:, :, :, 0].reshape(-1), [0.15, 0.50, 0.85])

    A_c = np.zeros(C); h_c = np.zeros(C); hd_c = np.zeros(C)
    fit_list = []; meta_list = []
    for c in range(C):
        xlag = dat["x_lag"][c].reshape(-1, 3)
        xcflag = dat["xcf_lag"][c].reshape(-1, 3)
        y = dat["y"][c].reshape(-1)
        pop = np.repeat(dat["pop"][c], T)
        offset = np.log(pop)
        region = np.repeat(np.arange(R), T)
        month = np.tile(np.arange(T) % 12, R)
        trend = np.tile(np.arange(T) / 12.0, R)
        X = build_X(xlag, region, month, trend, knots, R)
        fit = fit_glm(y, X, offset, region, cluster2=np.tile(np.arange(T) // 12, R))
        Xcf = cf_X(X, xcflag, knots, cfg.bias)
        mu = np.exp(np.clip(X @ fit["beta"] + offset, -30, 30))
        mucf = np.exp(np.clip(Xcf @ fit["beta"] + offset, -30, 30))
        A_c[c] = mu.sum() - mucf.sum()
        g = X.T @ mu - Xcf.T @ mucf
        h_c[c] = float(np.sqrt(max(g @ fit["V"] @ g, 0.0)))
        X2 = cf_X(X, xcflag, knots, cfg.bias + 0.01)
        mucf2 = np.exp(np.clip(X2 @ fit["beta"] + offset, -30, 30))
        hd_c[c] = float((mucf2.sum() - mucf.sum()) / 0.01)
        fit_list.append(fit)
        meta_list.append(dict(X=X, Xcf=Xcf, offset=offset, region=region, y=y, xcflag=xcflag))

    varD_c = (hd_c**2) * cfg.sd_gcm**2
    A_hat = float(A_c.sum())
    h_tot = float(np.sqrt((h_c**2).sum() + varD_c.sum()))
    G = min(f["G"] for f in fit_list)
    qt = tq(max(G - 1, 3))

    out = dict(cfg=cfg.label, seed=seed, A_true=dat["A_true"], A_true_c=dat["A_true_c"],
               A_hat=A_hat, h_tot=h_tot, h_c=h_c, hd_c=hd_c, A_c=A_c)

    # ---------------- Delta ----------------
    out["D"] = (A_hat - qt * h_tot, A_hat + qt * h_tot)
    out["D_c"] = np.column_stack([A_c - qt * np.sqrt(h_c**2 + varD_c), A_c + qt * np.sqrt(h_c**2 + varD_c)])

    # ---------------- Bootstrap ----------------
    boots = np.zeros((B, C))
    for c in range(C):
        fit = fit_list[c]; meta = meta_list[c]
        X = meta["X"]; off = meta["offset"]; reg = meta["region"]
        mu = fit["mu"]
        e = (meta["y"] - mu).reshape(R, T)
        Ainv = fit["Ainv"]
        for b in range(B):
            perm = rng.integers(0, R, R)
            estar = e[perm].reshape(-1)
            if boot == "score":
                beta_b = fit["beta"] + Ainv @ (X.T @ estar)
            else:
                ystar = np.maximum(np.round(mu + estar), 0.0)
                fb = fit_glm(ystar, X, off, reg, beta0=fit["beta"], iters=6)
                beta_b = fb["beta"]
            mub = np.exp(np.clip(X @ beta_b + off, -30, 30))
            mucfb = np.exp(np.clip(meta["Xcf"] @ beta_b + off, -30, 30))
            boots[b, c] = mub.sum() - mucfb.sum()
    boots = boots + rng.normal(0.0, 1.0, boots.shape) * np.sqrt(varD_c)[None, :]
    btot = boots.sum(1)
    out["B"] = (float(np.quantile(btot, alpha / 2)), float(np.quantile(btot, 1 - alpha / 2)))
    out["B_c"] = np.column_stack([np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)])
    out["boot_sd"] = float(np.std(btot))

    # ---------------- Bayesian (Laplace + RW2) ----------------
    D2 = np.array([[1.0, -1.0]])
    penb = 0.5 * (D2.T @ D2)
    draws = np.zeros((1000, C))
    for c in range(C):
        fit = fit_list[c]; meta = meta_list[c]
        X = meta["X"]; off = meta["offset"]
        p = X.shape[1]
        pen = np.eye(p) * 1e-4
        for l in range(3):
            s = 2 + 2 * l
            pen[s : s + 2, s : s + 2] += penb
        fb = fit_glm(meta["y"], X, off, meta["region"], beta0=fit["beta"], pen=pen, iters=15)
        w = np.maximum(fb["mu"], 1e-8)
        Sigma = fit["phi"] * np.linalg.inv(X.T @ (w[:, None] * X) + pen)
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
        bd = fb["beta"][None, :] + rng.normal(size=(1000, p)) @ L.T
        muf = np.exp(np.clip(bd @ X.T + off, -30, 30))
        mucf = np.exp(np.clip(bd @ meta["Xcf"].T + off, -30, 30))
        draws[:, c] = muf.sum(1) - mucf.sum(1)
    draws = draws + rng.normal(0.0, 1.0, draws.shape) * np.sqrt(varD_c)[None, :]
    dt = draws.sum(1)
    out["Y"] = (float(np.quantile(dt, alpha / 2)), float(np.quantile(dt, 1 - alpha / 2)))
    out["Y_c"] = np.column_stack([np.quantile(draws, alpha / 2, axis=0), np.quantile(draws, 1 - alpha / 2, axis=0)])

    # ---------------- Conformal ----------------
    # (a) naive split conformal: region-month residual scores (assumes iid units)
    z_units = []
    for c in range(C):
        fit = fit_list[c]; meta = meta_list[c]
        mu_all = fit["mu"].reshape(R, T)
        y = meta["y"].reshape(R, T)
        for r in range(R):
            z_units.extend((np.abs(y[r] - mu_all[r]) / np.sqrt(np.maximum(fit["phi"] * mu_all[r], 1e-8))).tolist())
    q_split = conformal_q(np.asarray(z_units), alpha)

    # exposure importance weights (counterfactual vs factual distribution)
    x3 = dat["x_lag"][:, :, :, 0].reshape(C, R, T)
    x3cf = dat["xcf_lag"][:, :, :, 0].reshape(C, R, T)
    mu_fact_mean = float(x3.mean())
    mu_cf_mean = float(x3cf.mean())
    sd_x = float(x3.std()) + 1e-9
    Wimp = np.exp(-0.5 * ((x3 - mu_cf_mean) / sd_x) ** 2 + 0.5 * ((x3 - mu_fact_mean) / sd_x) ** 2)
    Wimp = np.clip(Wimp / Wimp.mean(), 0.5, 2.0)
    ess = float(Wimp.sum() ** 2 / (Wimp**2).sum())

    # (b) block / weighted conformal: year-block split-half calibration of the attribution contrast
    years = np.tile(np.arange(T) // 12, R)
    z_blk = []; z_w = []; z_ctry = []
    KSPLIT = ksplit
    for k in range(KSPLIT):
        keep = rng.permutation(10)[:5]
        m1 = np.isin(years, keep)
        m2 = ~m1
        A1 = A2 = v1 = v2 = 0.0
        Aw1 = Aw2 = vw1 = vw2 = 0.0
        for c in range(C):
            fit = fit_list[c]; meta = meta_list[c]
            X = meta["X"]; off = meta["offset"]; reg = meta["region"]; y = meta["y"]
            f1 = fit_glm(y[m1], X[m1], off[m1], reg[m1], beta0=fit["beta"], iters=12, cluster2=np.tile(np.arange(T) // 12, R)[m1])
            f2 = fit_glm(y[m2], X[m2], off[m2], reg[m2], beta0=fit["beta"], iters=12, cluster2=np.tile(np.arange(T) // 12, R)[m2])
            reg_eff = np.zeros(R)
            for r in range(1, R):
                reg_eff[r] = fit["beta"][19 + r - 1]
            off_m = off.reshape(R, T)
            Xc = X[:, :19]
            Xcfc = meta["Xcf"][:, :19]
            wc = Wimp[c].reshape(-1)
            a1c = v1c = a2c = v2c = 0.0
            for f, tag in ((f1, 1), (f2, 2)):
                eta = (Xc @ f["beta"][:19]).reshape(R, T) + reg_eff[:, None] + off_m
                ecf = (Xcfc @ f["beta"][:19]).reshape(R, T) + reg_eff[:, None] + off_m
                mf = np.exp(eta)
                mc = np.exp(ecf)
                d = (mf - mc).reshape(-1)
                A = float(d.sum())
                g = Xc.T @ mf.reshape(-1) - Xcfc.T @ mc.reshape(-1)
                v = float(g @ f["V"][:19, :19] @ g)
                Aw = float((wc * d).sum())
                gw = Xc.T @ (wc * mf.reshape(-1)) - Xcfc.T @ (wc * mc.reshape(-1))
                vw = float(gw @ f["V"][:19, :19] @ gw)
                if tag == 1:
                    A1 += A; v1 += v; Aw1 += Aw; vw1 += vw
                    a1c += A; v1c += v
                else:
                    A2 += A; v2 += v; Aw2 += Aw; vw2 += vw
                    a2c += A; v2c += v
            z_ctry.append(abs(a1c - a2c) / np.sqrt(max(v1c + v2c, 1e-12)))
        z_blk.append(abs(A1 - A2) / np.sqrt(max(v1 + v2, 1e-12)))
        z_w.append(abs(Aw1 - Aw2) / np.sqrt(max(vw1 + vw2, 1e-12)))

    z_blk = np.asarray(z_blk)
    q_block = conformal_q(z_blk, alpha)
    q_w = conformal_q(np.asarray(z_w), alpha)
    q_ctry = conformal_q(np.asarray(z_ctry), alpha)

    out["CS"] = (A_hat - q_split * h_tot, A_hat + q_split * h_tot)
    out["CB"] = (A_hat - q_block * h_tot, A_hat + q_block * h_tot)
    out["CW"] = (A_hat - q_w * h_tot, A_hat + q_w * h_tot)
    out["CC"] = (A_hat - q_ctry * h_tot, A_hat + q_ctry * h_tot)
    hc_tot = np.sqrt(h_c**2 + varD_c)
    out["CS_c"] = np.column_stack([A_c - q_split * hc_tot, A_c + q_split * hc_tot])
    out["CB_c"] = np.column_stack([A_c - q_block * hc_tot, A_c + q_block * hc_tot])
    out["CW_c"] = np.column_stack([A_c - q_w * hc_tot, A_c + q_w * hc_tot])
    out["CC_c"] = np.column_stack([A_c - q_ctry * hc_tot, A_c + q_ctry * hc_tot])
    out["q"] = np.array([q_split, q_block, q_w])
    out["ess"] = ess
    return out
