"""Known-truth ancillary decomposition on a fixed real exposure design."""
from __future__ import annotations

import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(case_root):
    config_file = case_root / 'inputs/real_config.json'
    config = json.loads(config_file.read_text(encoding='utf-8-sig'))
    code_file = case_root / 'code/real_panel_audit.py'
    spec = importlib.util.spec_from_file_location('diagnostic_case_model', code_file)
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)

    artifact = case_root / 'results/real_design.npz'
    # This project-generated artifact stores its date vector as an object array.
    archive = np.load(artifact, allow_pickle=True)
    X, Xcf = archive['X'], archive['Xcf']
    offset, beta, observed = archive['offset'], archive['beta'], archive['y']
    dates = pd.to_datetime(archive['outcome_dates'])
    panel = pd.read_parquet(case_root / config['source']['aggregate_panel'])
    selected = panel.loc[panel.state.astype(str).eq(str(config['design']['state_code']))]
    temperature = selected.set_index('date').tmean.reindex(dates).to_numpy()
    assert np.isfinite(temperature).all()
    compatible_mu = np.exp(X @ beta + offset)
    compatible_cf = np.exp(Xcf @ beta + offset)
    amplitude = config['simulation']['omitted_interaction']['amplitude']
    interaction = model._interaction_component(dates, temperature, amplitude)
    interaction_cf = model._interaction_component(
        dates, temperature - config['estimand']['counterfactual_shift_c'], amplitude)
    scenario_means = {
        'model_compatible': (compatible_mu, compatible_cf),
        'smooth_omitted_interaction': (compatible_mu * np.exp(interaction), compatible_cf * np.exp(interaction_cf)),
        'correlated_overdispersed_ar1': (compatible_mu, compatible_cf)
    }
    sigma2 = model._lognormal_variance_from_residuals(observed, compatible_mu)
    phi = config['simulation']['correlated_overdispersed']['phi']
    rep_file = case_root / 'results/real_formal_replications.csv'
    replications = pd.read_csv(rep_file)
    z = norm.ppf(1 - config['inference']['alpha'] / 2)
    rows, interval_rows = [], []

    for name, (mu_true, cf_true) in scenario_means.items():
        pseudo = model.fit_poisson_irls(X, mu_true, offset, Xcf=Xcf, hac_bandwidths=(14,),
                                       max_iterations=100, tolerance=1e-10)
        assert pseudo['converged']
        mu_pseudo = pseudo['mu']
        cf_pseudo = pseudo['mu_cf']
        true_a = float(np.sum(mu_true - cf_true))
        pseudo_a = float(np.sum(mu_pseudo - cf_pseudo))
        bias = pseudo_a - true_a
        gradient = X.T @ mu_pseudo - Xcf.T @ cf_pseudo
        information = X.T @ (mu_pseudo[:, None] * X)
        influence = X @ np.linalg.solve(information, gradient)
        variance = float(np.sum(influence ** 2 * mu_true))
        used_sigma2 = sigma2 if name == 'correlated_overdispersed_ar1' else 0.
        if used_sigma2 > 0:
            weighted = influence * mu_true
            variance += float(np.dot(weighted, weighted) * np.expm1(used_sigma2))
            for lag in range(1, len(weighted)):
                kernel = np.expm1(used_sigma2 * phi ** lag)
                if kernel == 0:
                    break
                variance += float(2 * kernel * np.dot(weighted[:-lag], weighted[lag:]))
        assert variance > 0
        oracle_se = float(np.sqrt(variance))
        planned = replications.loc[replications.scenario.eq(name)].copy()
        valid = planned.loc[planned.converged.astype(bool)].copy()
        assert np.allclose(valid.A_true, true_a, rtol=1e-10, atol=1e-7)
        error = valid.A_hat.to_numpy() - true_a
        row = {
            'scenario': name, 'n_planned': len(planned), 'n_converged': len(valid),
            'A_true': true_a, 'pseudo_true_A': pseudo_a, 'deterministic_projection_bias': bias,
            'empirical_mean_error': float(error.mean()), 'empirical_error_sd': float(error.std(ddof=1)),
            'mcse_mean_error': float(error.std(ddof=1) / np.sqrt(len(error))),
            'known_process_linearized_oracle_se': oracle_se,
            'ar1_sigma2': used_sigma2, 'ar1_phi': phi if used_sigma2 > 0 else 0.,
            'expected_score_max_abs': float(np.max(np.abs(X.T @ (mu_true - mu_pseudo))))
        }
        for method in ['model', 'hac14', 'hac28']:
            row[f'mean_se_{method}'] = float(valid[f'se_{method}'].mean())
            row[f'mean_se_over_empirical_sd_{method}'] = float(valid[f'se_{method}'].mean() / error.std(ddof=1))
        for label, centering in [('oracle_scale', 0.), ('oracle_scale_and_known_bias', bias)]:
            centers = valid.A_hat.to_numpy() - centering
            covered = np.abs(centers - true_a) <= z * oracle_se
            count = int(covered.sum())
            n = len(planned)
            p = count / n
            row[f'coverage_{label}'] = p
            row[f'mcse_coverage_{label}'] = float(np.sqrt(p * (1 - p) / n))
            row[f'width_{label}'] = float(2 * z * oracle_se)
            for replicate, is_covered in zip(valid.replicate, covered):
                interval_rows.append({'scenario': name, 'replicate': int(replicate), 'method': label,
                                      'covered': bool(is_covered), 'width': float(2 * z * oracle_se)})
        rows.append(row)

    output = output_path(case_root / 'results/mechanism_diagnostic.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(interval_rows).to_csv(output_path(case_root / 'results/mechanism_diagnostic_intervals.csv'), index=False)
    manifest = {
        'analysis_role': 'Ancillary known-truth mechanism diagnostic added after the initial state formal summary was inspected.',
        'pseudo_true_definition': 'Deterministic Poisson-score projection of each known generating mean onto the fixed fitted design.',
        'covariance_definition': 'Known conditional count covariance for independent Poisson or the declared stationary lognormal-Poisson AR1 process.',
        'oracle_interpretation': 'Linearized sampling-scale and truth-known centering benchmarks for mechanism interpretation.',
        'primary_coverage_tables': 'The original formal method comparisons retain their registered definitions.',
        'input_hashes': {str(path.relative_to(case_root)): digest(path) for path in [artifact, config_file, code_file, rep_file]},
        'diagnostic_code_sha256': digest(Path(__file__)), 'output_sha256': digest(output),
        'rows': rows
    }
    output_path(case_root / 'results/mechanism_diagnostic.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-root', type=Path, default=Path(__file__).resolve().parents[1])
    analyze(parser.parse_args().case_root)
