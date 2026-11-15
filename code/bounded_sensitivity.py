"""Finite-grid sensitivity bounds for an explicitly mapped lag-spline class.

The routine deliberately does not infer coefficient positions. Callers must pass
an exposure-block metadata mapping for the exact design they constructed.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError


def _metadata(meta: Mapping[str, Any], p: int):
    if not isinstance(meta, Mapping):
        raise TypeError("block_metadata must be a mapping with explicit lag_blocks")
    required = ("design_source", "basis_names", "lag_blocks")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"block_metadata is missing required keys: {missing}")
    source = str(meta["design_source"]).strip()
    if not source:
        raise ValueError("block_metadata.design_source must identify the design mapping")
    basis_names = tuple(str(x) for x in meta["basis_names"])
    if len(basis_names) != 2 or any(not x for x in basis_names):
        raise ValueError("this implementation requires two named spline columns per lag")
    raw_blocks = meta["lag_blocks"]
    if not raw_blocks:
        raise ValueError("at least one explicitly mapped lag block is required")
    blocks, names, used = [], [], []
    for item in raw_blocks:
        if not isinstance(item, Mapping) or "name" not in item or "columns" not in item:
            raise ValueError("each lag block must contain explicit 'name' and 'columns'")
        name = str(item["name"]).strip()
        raw_cols = np.asarray(item["columns"], dtype=object)
        if not name or raw_cols.shape != (2,):
            raise ValueError("each lag block needs a name and exactly two column indices")
        if any(isinstance(value, (bool, np.bool_)) for value in raw_cols):
            raise ValueError(f"lag block {name!r} column indices must be integers, not booleans")
        try:
            numeric_cols = np.asarray(item["columns"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lag block {name!r} column indices must be integers") from exc
        if not np.all(np.isfinite(numeric_cols)) or not np.all(numeric_cols == np.floor(numeric_cols)):
            raise ValueError(f"lag block {name!r} column indices must be finite integers")
        cols = numeric_cols.astype(int)
        if name in names:
            raise ValueError(f"lag block names must be unique; repeated name: {name!r}")
        if np.any(cols < 0) or np.any(cols >= p):
            raise ValueError(f"lag block {name!r} has a column outside the design")
        if len(set(cols.tolist())) != 2:
            raise ValueError(f"lag block {name!r} repeats a column")
        if set(cols.tolist()) & set(used):
            raise ValueError("lag blocks must be disjoint")
        blocks.append(cols)
        names.append(name)
        used.extend(cols.tolist())
    return source, basis_names, tuple(names), tuple(blocks), np.asarray(used, dtype=int)


def _finite_exp(eta: np.ndarray, label: str) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if not np.all(np.isfinite(eta)):
        raise ValueError(f"{label} linear predictors contain non-finite values")
    if np.max(eta, initial=-np.inf) >= np.log(np.finfo(float).max) - 2.0:
        raise OverflowError(f"{label} linear predictor is too large for a finite exponential bound")
    with np.errstate(under="ignore"):
        out = np.exp(eta)
    if not np.all(np.isfinite(out)):
        raise OverflowError(f"{label} means are non-finite")
    return out


def _vertices_from_symmetric_slabs(C: np.ndarray, delta: float, tol: float):
    """Return vertices of {u: |C u| <= delta}; use 2-D hull to prune redundant slabs."""
    normals = np.vstack((C, -C))
    normals = np.unique(normals, axis=0)
    try:
        hull = ConvexHull(normals)
    except QhullError as exc:
        raise RuntimeError("could not certify the two-dimensional constraint hull") from exc
    A = normals[hull.vertices]
    b = np.full(len(A), delta, dtype=float)
    found: list[np.ndarray] = []
    for i, j in combinations(range(len(A)), 2):
        mat = np.vstack((A[i], A[j]))
        scale = np.linalg.norm(A[i]) * np.linalg.norm(A[j])
        det = np.linalg.det(mat)
        if scale == 0 or abs(det) <= 32.0 * np.finfo(float).eps * scale:
            continue
        u = np.linalg.solve(mat, np.array([delta, delta]))
        if np.all(A @ u <= delta + tol * (1.0 + abs(delta) + np.linalg.norm(u))):
            if not any(np.linalg.norm(u - v, ord=np.inf) <= tol * (1.0 + np.linalg.norm(u, ord=np.inf)) for v in found):
                found.append(u)
    if not found:
        raise RuntimeError("bounded feasible 2-D class yielded no enumerated vertices")
    V = np.asarray(found, dtype=float)
    # Check against the complete, unpruned system: hull pruning must preserve every slab.
    violation = float(np.max(np.abs(C @ V.T) - delta, initial=-np.inf))
    allowed = tol * (1.0 + abs(delta) + float(np.max(np.linalg.norm(V, axis=1))))
    if violation > allowed:
        raise RuntimeError(f"enumerated vertices violate an original slab by {violation:g}")
    return V, A, b


def _maximize(direction: np.ndarray, A: np.ndarray, b: np.ndarray):
    res = linprog(-direction, A_ub=A, b_ub=b, bounds=[(None, None), (None, None)], method="highs")
    if res.status == 3:
        return {"status": "unbounded", "value": None, "u": None, "message": res.message}
    if res.status == 2:
        return {"status": "infeasible", "value": None, "u": None, "message": res.message}
    if not res.success:
        return {"status": "solver_error", "value": None, "u": None, "message": res.message}
    return {"status": "optimal", "value": float(direction @ res.x), "u": res.x.astype(float), "message": res.message}


def _rowwise_vertex_max(D: np.ndarray, V: np.ndarray, chunk_rows: int = 4096):
    out = np.empty(D.shape[0], dtype=float)
    for start in range(0, D.shape[0], chunk_rows):
        stop = min(start + chunk_rows, D.shape[0])
        out[start:stop] = np.max(np.abs(D[start:stop] @ V.T), axis=1)
    return out


def _class_status(C: np.ndarray, delta: float, tol: float):
    # The feasible set always contains zero when delta >= 0. Rank(C)=2 is
    # necessary and sufficient for boundedness of this symmetric slab class.
    rank = int(np.linalg.matrix_rank(C)) if C.size else 0
    if rank < 2:
        H = np.vstack((C, -C)) if C.size else np.zeros((0, 2))
        b = np.full(H.shape[0], delta, dtype=float)
        return {"feasible": True, "bounded": False, "rank": rank, "A": H, "b": b, "vertices": None}
    V, A, b = _vertices_from_symmetric_slabs(C, delta, tol)
    return {"feasible": True, "bounded": True, "rank": rank, "A": A, "b": b, "vertices": V}


def unrestricted_cumulative_nullspace_diagnostic(
    beta: Sequence[float], X: np.ndarray, Xcf: np.ndarray, offset: Sequence[float],
    block_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Exhibit the two-lag zero-sum direction and its attribution-gradient slope."""
    beta = np.asarray(beta, dtype=float)
    X, Xcf = np.asarray(X, float), np.asarray(Xcf, float)
    offset = np.asarray(offset, dtype=float)
    source, basis_names, lag_names, blocks, _ = _metadata(block_metadata, X.shape[1])
    if len(blocks) != 2:
        raise ValueError("the nullspace diagnostic is defined for exactly two lag blocks")
    if X.shape != Xcf.shape or beta.shape != (X.shape[1],) or offset.shape != (X.shape[0],):
        raise ValueError("beta, X, Xcf, and offset dimensions do not align")
    mu = _finite_exp(X @ beta + offset, "factual")
    mucf = _finite_exp(Xcf @ beta + offset, "counterfactual")
    g0 = X[:, blocks[0]].T @ mu - Xcf[:, blocks[0]].T @ mucf
    g1 = X[:, blocks[1]].T @ mu - Xcf[:, blocks[1]].T @ mucf
    diff = g0 - g1
    norm = float(np.linalg.norm(diff))
    direction = (diff / norm) if norm > 0 else np.zeros(2)
    slope = float(diff @ direction)
    return {
        "class": "unrestricted cumulative-only two-lag perturbations",
        "design_source": source,
        "lag_names": list(lag_names),
        "basis_names": list(basis_names),
        "direction_lag0": direction.tolist(),
        "direction_lag1": (-direction).tolist(),
        "cumulative_direction_max_abs": float(np.max(np.abs(direction - direction))),
        "gradient_slope_per_unit_scale": slope,
        "gradient_difference_norm": norm,
        "unbounded_linear_objective_demonstrated": bool(norm > 1e-12),
        "proof": "For d_beta0=v and d_beta1=-v, the cumulative coefficient perturbation is zero and the linear attribution change is (g0-g1)'v. Scaling v gives an unbounded linear objective whenever g0 != g1.",
    }


def bounded_sensitivity(
    beta: Sequence[float],
    X: np.ndarray,
    Xcf: np.ndarray,
    offset: Sequence[float],
    block_metadata: Mapping[str, Any],
    weights: Sequence[float],
    delta: float = 0.05,
    base_half_width: float = 0.0,
    *,
    sample_count: int = 64,
    seed: int = 20261112,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Bound attribution sensitivity for a fixed-weight, finite-grid lag class.

    Perturbation class: d_beta_l = weights[l] * u, weights >= 0 and sum to 1.
    Constraints: |b(t)'u| <= delta at every factual and counterfactual basis
    row supplied by every explicitly mapped lag block. This is a finite-grid
    statement over observed design rows, not a continuous-temperature claim.
    """
    beta = np.asarray(beta, dtype=float)
    X, Xcf = np.asarray(X, dtype=float), np.asarray(Xcf, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if X.ndim != 2 or Xcf.shape != X.shape:
        raise ValueError("X and Xcf must be same-shaped 2-D arrays")
    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError("X and Xcf must contain at least one row and one column")
    if beta.shape != (p,) or offset.shape != (n,):
        raise ValueError("beta and offset must align with design dimensions")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Xcf)) and np.all(np.isfinite(beta)) and np.all(np.isfinite(offset))):
        raise ValueError("inputs must be finite")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if not np.isfinite(base_half_width) or base_half_width < 0:
        raise ValueError("base_half_width must be finite and nonnegative")
    if isinstance(sample_count, (bool, np.bool_)) or not isinstance(sample_count, (int, np.integer)) or sample_count < 0:
        raise ValueError("sample_count must be a nonnegative integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    source, basis_names, lag_names, blocks, used_cols = _metadata(block_metadata, p)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(blocks),) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must have one finite value per mapped lag block")
    if np.any(weights < 0) or not np.isclose(float(weights.sum()), 1.0, rtol=0, atol=1e-12):
        raise ValueError("allocation weights must be nonnegative and sum to one")

    # D matrices map u to each row's factual/counterfactual linear-predictor change.
    Df = np.zeros((n, 2), dtype=float)
    Dcf = np.zeros((n, 2), dtype=float)
    support_basis = []
    for weight, cols in zip(weights, blocks):
        Df += weight * X[:, cols]
        Dcf += weight * Xcf[:, cols]
        support_basis.extend((X[:, cols], Xcf[:, cols]))
    C = np.vstack(support_basis)
    class_info = _class_status(C, float(delta), tol)
    A_ub, b_ub = class_info["A"], class_info["b"]

    eta_f = X @ beta + offset
    eta_cf = Xcf @ beta + offset
    mu, mucf = _finite_exp(eta_f, "factual"), _finite_exp(eta_cf, "counterfactual")
    A0 = float(mu.sum() - mucf.sum())
    gradient = Df.T @ mu - Dcf.T @ mucf
    max_pos = _maximize(gradient, A_ub, b_ub)
    max_neg = _maximize(-gradient, A_ub, b_ub)
    lp_ok = max_pos["status"] == "optimal" and max_neg["status"] == "optimal"
    B_lin = max(float(max_pos["value"]), float(max_neg["value"])) if lp_ok else None

    result: dict[str, Any] = {
        "method": "fixed_weight_lag_allocation_finite_grid_v1",
        "design_source": source,
        "lag_names": list(lag_names),
        "lag_block_columns": [cols.tolist() for cols in blocks],
        "basis_names": list(basis_names),
        "allocation_weights": weights.tolist(),
        "delta": float(delta),
        "constraint_grid": "All observed factual and counterfactual basis rows for every explicitly mapped lag block; finite support only.",
        "constraint_rows": int(C.shape[0]),
        "constraint_unique_basis_rows": int(np.unique(C, axis=0).shape[0]),
        "feasible": class_info["feasible"],
        "bounded_class": class_info["bounded"],
        "constraint_rank": class_info["rank"],
        "A_hat": A0,
        "gradient_u": gradient.tolist(),
        "linear_max_plus": {**max_pos, "u": max_pos["u"].tolist() if max_pos["u"] is not None else None},
        "linear_max_minus": {**max_neg, "u": max_neg["u"].tolist() if max_neg["u"] is not None else None},
        "B_lin": B_lin,
        "base_half_width": float(base_half_width),
        "exposure_columns_only_perturbed": True,
        "nuisance_column_count": int(p - len(used_cols)),
        "continuous_temperature_certificate": False,
        "class_scope": "conditional sensitivity bound for the stated finite-grid and fixed-allocation perturbation class",
    }
    if not class_info["bounded"]:
        result.update({
            "R_uniform": None,
            "sensitivity_radius": None,
            "A_sensitivity_bounds": None,
            "interval_bounds": None,
            "threshold_diagnostics": {"status": "unbounded_class_no_full_class_sensitivity_claim"},
            "adversarial_sample_checks": {"status": "not_run_unbounded_class"},
        })
        return result
    if not lp_ok:
        result.update({
            "R_uniform": None,
            "sensitivity_radius": None,
            "A_sensitivity_bounds": None,
            "interval_bounds": None,
            "threshold_diagnostics": {"status": "linear_program_failed", "statuses": [max_pos["status"], max_neg["status"]]},
            "adversarial_sample_checks": {"status": "not_run_solver_failure"},
        })
        return result

    V = class_info["vertices"]
    # Row-level max absolute predictor changes are exact for a bounded polygon.
    rf = _rowwise_vertex_max(Df, V)
    rcf = _rowwise_vertex_max(Dcf, V)
    with np.errstate(over="ignore", invalid="ignore"):
        rem_f = np.expm1(rf) - rf
        rem_cf = np.expm1(rcf) - rcf
        R_uniform = float(mu @ rem_f + mucf @ rem_cf)
    if not np.isfinite(R_uniform):
        R_uniform = float("inf")
    sensitivity_radius = float(B_lin + R_uniform)
    sens_lo, sens_hi = float(A0 - sensitivity_radius), float(A0 + sensitivity_radius)
    interval_lo = float(A0 - base_half_width - sensitivity_radius)
    interval_hi = float(A0 + base_half_width + sensitivity_radius)

    # Verify the LP extrema against the independently enumerated polygon vertices.
    vertex_plus = float(np.max(V @ gradient))
    vertex_minus = float(np.max(V @ (-gradient)))
    lp_vertex_error = max(abs(vertex_plus - max_pos["value"]), abs(vertex_minus - max_neg["value"]))
    candidates = [v.copy() for v in V]
    candidates.extend([max_pos["u"].copy(), max_neg["u"].copy()])
    if sample_count:
        rng = np.random.default_rng(seed)
        for _ in range(sample_count):
            mix = rng.dirichlet(np.ones(len(V)))
            candidates.append(mix @ V)
    candidate_array = np.asarray(candidates, dtype=float)
    max_constraint_violation = float(np.max(np.abs(C @ candidate_array.T) - delta, initial=-np.inf))
    allowed = tol * (1.0 + abs(delta) + float(np.max(np.linalg.norm(candidate_array, axis=1))))
    all_feasible = bool(max_constraint_violation <= allowed)
    max_abs_actual_shift = 0.0
    max_abs_actual_remainder = 0.0
    max_mapping_error = 0.0
    max_nuisance_change = 0.0
    remainder_samples_within_uniform = True
    attribution_samples_within_envelope = True
    for u in candidate_array:
        dbeta = np.zeros(p, dtype=float)
        for weight, cols in zip(weights, blocks):
            dbeta[cols] = weight * u
        if len(used_cols) < p:
            max_nuisance_change = max(max_nuisance_change, float(np.max(np.abs(dbeta[np.setdiff1d(np.arange(p), used_cols)]), initial=0.0)))
        direct_df = X @ dbeta
        direct_dcf = Xcf @ dbeta
        max_mapping_error = max(max_mapping_error, float(np.max(np.abs(direct_df - Df @ u), initial=0.0)), float(np.max(np.abs(direct_dcf - Dcf @ u), initial=0.0)))
        with np.errstate(over="ignore", invalid="ignore"):
            Af = float(np.sum(mu * np.exp(direct_df)))
            Acf = float(np.sum(mucf * np.exp(direct_dcf)))
        actual_shift = (Af - Acf) - A0
        linear_shift = float(gradient @ u)
        remainder = actual_shift - linear_shift
        max_abs_actual_shift = max(max_abs_actual_shift, abs(actual_shift))
        max_abs_actual_remainder = max(max_abs_actual_remainder, abs(remainder))
        if abs(remainder) > R_uniform + tol * (1.0 + abs(R_uniform)):
            remainder_samples_within_uniform = False
        if abs(actual_shift) > sensitivity_radius + tol * (1.0 + sensitivity_radius):
            attribution_samples_within_envelope = False
    checks_ok = bool(all_feasible and remainder_samples_within_uniform and attribution_samples_within_envelope and max_nuisance_change == 0.0 and max_mapping_error <= 1e-9 and lp_vertex_error <= 1e-7 * (1 + abs(B_lin)))

    if interval_lo > 0:
        threshold_status = "positive_exclusion"
    elif interval_hi < 0:
        threshold_status = "negative_exclusion"
    else:
        threshold_status = "zero_not_excluded"
    result.update({
        "polytope_vertex_count": int(len(V)),
        "B_lin_vertex_check": float(max(vertex_plus, vertex_minus)),
        "LP_vertex_objective_max_error": float(lp_vertex_error),
        "rowwise_predictor_max_abs_factual": float(np.max(rf, initial=0.0)),
        "rowwise_predictor_max_abs_counterfactual": float(np.max(rcf, initial=0.0)),
        "R_uniform": R_uniform,
        "sensitivity_radius": sensitivity_radius,
        "A_sensitivity_bounds": [sens_lo, sens_hi],
        "interval_bounds": [interval_lo, interval_hi],
        "threshold_diagnostics": {
            "status": threshold_status,
            "base_interval_half_width": float(base_half_width),
            "expanded_interval_excludes_zero": bool(interval_lo > 0 or interval_hi < 0),
            "signed_margin_from_zero": float(interval_lo if interval_lo > 0 else interval_hi if interval_hi < 0 else 0.0),
        },
        "adversarial_sample_checks": {
            "status": "pass" if checks_ok else "fail",
            "sample_scope": "all enumerated vertices, both LP optimizers, and seeded convex combinations; diagnostics only, with the row-wise remainder supplying the class-wide argument",
            "candidate_count": int(len(candidate_array)),
            "max_constraint_violation": max_constraint_violation,
            "all_candidates_feasible": all_feasible,
            "max_abs_actual_attribution_shift": max_abs_actual_shift,
            "max_abs_actual_remainder": max_abs_actual_remainder,
            "actual_remainders_within_R_uniform": remainder_samples_within_uniform,
            "actual_shifts_within_sensitivity_envelope": attribution_samples_within_envelope,
            "max_nuisance_coefficient_change": max_nuisance_change,
            "max_design_mapping_reconstruction_error": max_mapping_error,
            "LP_vs_vertex_objective_error": float(lp_vertex_error),
            "all_checks_pass": checks_ok,
        },
    })
    return result


def _toy_basis(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # Two explicit basis columns; the finite-grid logic does not depend on
    # the particular basis parameterization used by the real design.
    return np.column_stack((x, np.maximum(x - 20.0, 0.0) ** 2 / 10.0))


def validate_toy() -> dict[str, Any]:
    """Deterministic 2-lag monthly-design validation, including nuisance columns."""
    R, T = 3, 12
    state = np.repeat(np.arange(R), T)
    month = np.tile(np.arange(T), R)
    time = np.tile(np.arange(T), R).astype(float) / 12.0
    x0 = 17.0 + 4.0 * np.sin(2.0 * np.pi * month / 12.0) + 0.35 * state
    x1 = 18.0 + 3.0 * np.cos(2.0 * np.pi * month / 12.0) + 0.20 * state
    xc0, xc1 = x0 - 0.8, x1 - 0.65
    # Match the source S1 column ordering: intercept, trend, two basis columns
    # per lag, eleven month indicators, then R-1 region indicators.
    month_fe = np.column_stack([(month == m).astype(float) for m in range(1, 12)])
    region_fe = np.column_stack([(state == r).astype(float) for r in range(1, R)])
    X = np.column_stack((np.ones(len(month)), time, _toy_basis(x0), _toy_basis(x1), month_fe, region_fe))
    Xcf = np.column_stack((np.ones(len(month)), time, _toy_basis(xc0), _toy_basis(xc1), month_fe, region_fe))
    beta = np.array([0.15, -0.01, 0.012, -0.003, 0.009, 0.005] + [0.002 * k for k in range(1, 12)] + [0.01, -0.008])
    offset = np.log(np.linspace(850.0, 1250.0, len(month)))
    metadata = {
        "design_source": "toy monthly S1 layout mirroring c07_bai_frozen.design; explicit mapping supplied by caller",
        "basis_names": ["linear_temperature", "upper_tail_quadratic"],
        "lag_blocks": [
            {"name": "lag0", "columns": [2, 3]},
            {"name": "lag1", "columns": [4, 5]},
        ],
    }
    result = bounded_sensitivity(beta, X, Xcf, offset, metadata, weights=[0.5, 0.5], delta=0.05, base_half_width=2.0, sample_count=80, seed=17)
    nullspace = unrestricted_cumulative_nullspace_diagnostic(beta, X, Xcf, offset, metadata)
    result["unrestricted_cumulative_nullspace_diagnostic"] = nullspace
    result["toy_metadata_check"] = {
        "design_columns": int(X.shape[1]),
        "month_fe_columns": list(range(6, 17)),
        "region_fe_columns": list(range(17, 19)),
        "mapped_exposure_columns": result["lag_block_columns"],
        "month_fe_perturbed": False,
        "region_fe_perturbed": False,
    }
    assert result["bounded_class"] and result["feasible"]
    assert result["adversarial_sample_checks"]["all_checks_pass"]
    assert result["toy_metadata_check"]["month_fe_perturbed"] is False
    assert result["toy_metadata_check"]["region_fe_perturbed"] is False
    assert nullspace["unbounded_linear_objective_demonstrated"]

    # A rank-deficient finite-grid class has a free coefficient direction and
    # must not be reported as a bounded sensitivity result.
    rank_deficient_X = X.copy()
    rank_deficient_Xcf = Xcf.copy()
    for cols in ([2, 3], [4, 5]):
        rank_deficient_X[:, cols[0]] = 1.0
        rank_deficient_X[:, cols[1]] = 0.0
        rank_deficient_Xcf[:, cols[0]] = 1.0
        rank_deficient_Xcf[:, cols[1]] = 0.0
    unbounded_class = bounded_sensitivity(
        beta, rank_deficient_X, rank_deficient_Xcf, offset, metadata,
        weights=[0.5, 0.5], delta=0.05, sample_count=0,
    )
    result["rank_deficient_class_check"] = {
        "bounded_class": bool(unbounded_class["bounded_class"]),
        "constraint_rank": int(unbounded_class["constraint_rank"]),
        "sensitivity_radius": unbounded_class["sensitivity_radius"],
        "threshold_status": unbounded_class["threshold_diagnostics"]["status"],
    }
    assert not unbounded_class["bounded_class"]
    assert unbounded_class["sensitivity_radius"] is None
    assert unbounded_class["threshold_diagnostics"]["status"] == "unbounded_class_no_full_class_sensitivity_claim"
    return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-check", action="store_true", help="run deterministic toy validation")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    if not args.toy_check:
        parser.error("--toy-check is required; real-data arrays must be supplied by a caller")
    report = validate_toy()
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
