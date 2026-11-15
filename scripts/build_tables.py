#!/usr/bin/env python3
"""Rebuild paper tables from archived replication-level results."""
from __future__ import annotations
import argparse, importlib.util, json, math, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "delta_corrected_cr1_tw", "delta_corrected_common_cr1_tw", "delta_legacy_covariance",
    "delta_pinned_source", "delta_region_cr1_context", "delta_independent_HC1_positive_control",
    "independent_scale_calibrated", "truth_trained_bias_centered_split_calibrated",
    "estimator_only_common_bias", "estimator_only_common_bias_bias_centered", "year_half_split_q_diagnostic",
]
COPY_TABLES = [
    "Table_S9_method_comparison.csv",
    "Table_S2_calibration_diagnostics.csv", "Table_S5_mechanism_oracle_diagnostics.csv",
    "Table_S6_monthly_finite_grid_bounds.csv", "Table_S7_data_dictionary.csv",
    "Table_S8_analytic_panel_descriptions.csv", "Table_S12_allocation_bounds.csv",
    "Table_S13a_in_class_checks.csv", "Table_S13b_out_of_class_checks.csv",
    "Table_S14_monthly_sampling_information.csv",
]
CHECK_KEYS = {
    "Table_S1_fixed_method_summaries.csv": ["scenario", "design_seed", "method", "family", "case"],
    "Table_S3_fixed_paired_differences.csv": ["scenario", "design_seed", "method", "family", "case"],
    "Table_S4a_daily_coverage.csv": ["case", "scenario", "method"],
    "Table_S4b_real_paired_differences.csv": ["scenario", "method_1", "method_2", "case"],
    "Table_S4c_quasi_paired_differences.csv": ["case", "scenario", "method_1", "method_2"],
    "Table_S10_mvn_summary.csv": ["case", "scenario", "covariance_method", "interval_method"],
    "Table_S11_mvn_method_paired.csv": ["case", "scenario", "covariance_method", "method_1", "method_2"],
    "Table_S11b_covariance_paired.csv": ["case", "scenario", "interval_method", "method_1", "method_2"],
    "Table_S11c_stability_summary.csv": ["case", "covariance_method", "interval_method", "n_draws"],
    "Table_S11d_stability_detailed.csv": ["case", "scenario", "replicate", "covariance_method", "interval_method", "n_draws"],
}
BOOLS = {"fit_ok", "fit_converged", "converged", "interval_valid", "covered", "covariance_psd_valid",
         "match_A_hat_1e_8", "match_se_model_1e_8"}

def flag(x):
    if pd.isna(x): return False
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, float, np.number)): return bool(x == 1)
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def read(path):
    if not path.is_file(): raise FileNotFoundError(f"required input missing: {path}")
    d = pd.read_csv(path)
    for c in d:
        if c in BOOLS or c.startswith("interval_valid__") or c.startswith("cover__") or c.startswith("covered_"):
            d[c] = d[c].map(flag)
    if "fit_flags" in d: d["fit_flags"] = d["fit_flags"].fillna("").astype(str)
    return d

def num(x): return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
def mcse(x):
    a = np.asarray(x, dtype=float); a = a[np.isfinite(a)]
    return float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else float("nan")

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    if s is None or s.loader is None: raise ImportError(f"cannot import {path}")
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m

def fixed_tables():
    fx = load("table_rebuild_fixed", ROOT/"code"/"fixed_design.py")
    hd = load("table_rebuild_heatday", ROOT/"code"/"fixed_heatday.py")
    summaries, pairs, sources = [], [], []
    for prefix, family, module, is_heatday in [
        ("fixed", "fixed_temperature", fx, False), ("heatday", "fixed_heatday", hd, True)
    ]:
        files = sorted((ROOT/"results").glob(f"{prefix}_formal_panel_*_seed*.csv"))
        if not files: raise FileNotFoundError(f"no formal {prefix} panels")
        for path in files:
            d = read(path)
            if "stage" not in d or not d.stage.astype(str).eq("test").any():
                raise ValueError(f"no test-stage records in {path}")
            d = d.loc[d.stage.astype(str).eq("test")]
            for (scenario, seed), g in d.sort_values("replicate").groupby(["scenario", "design_seed"], sort=False):
                rows = g.to_dict(orient="records")
                cfg = module.read_config()
                spec = cfg["year_split_diagnostic"]
                year_n = min(int(spec["formal_test_reps_per_panel"]), len(rows)) if spec.get("enabled", False) else 0
                for method in METHODS:
                    eval_rows = rows[:year_n] if method == "year_half_split_q_diagnostic" else rows
                    result = module.summarize_panel(rows, method, str(scenario), int(seed), eval_rows=eval_rows) if is_heatday else module.summarize(rows, method, str(scenario), int(seed), eval_rows=eval_rows)
                    result["mean_error_raw_fit"] = result.pop("mean_error")
                    result.update({"family": family, "case": "fixed_design"})
                    summaries.append(result)
                    ref = "delta_corrected_cr1_tw"
                    c1 = np.asarray([flag(r.get("cover__"+method, False)) for r in eval_rows], dtype=float)
                    c0 = np.asarray([flag(r.get("cover__"+ref, False)) for r in eval_rows], dtype=float)
                    w1 = num([r.get("width__"+method, np.nan) for r in eval_rows])
                    w0 = num([r.get("width__"+ref, np.nan) for r in eval_rows])
                    ok = np.isfinite(w1) & np.isfinite(w0); wd = w1[ok]-w0[ok]; diff=c1-c0
                    pairs.append({"scenario":scenario,"design_seed":int(seed),"method":method,"reference":ref,
                        "n_planned":len(eval_rows),"coverage_diff":float(diff.mean()),"coverage_diff_mcse_paired_1se":mcse(diff),
                        "n_width_pairs":int(ok.sum()),"n_width_infinite_method":int(np.isinf(w1).sum()),
                        "n_width_infinite_reference":int(np.isinf(w0).sum()),
                        "mean_width_diff":float(wd.mean()) if len(wd) else float("nan"),
                        "width_diff_mcse_paired_1se":mcse(wd),"family":family,"case":"fixed_design"})
            sources.append(str(path.relative_to(ROOT)))
    return pd.DataFrame(summaries), pd.DataFrame(pairs), sources

def daily_tables():
    covrows, realpairs, quasipairs, sources = [], [], [], []
    for label, folder in [("state",ROOT/"results"),("municipality",ROOT/"city_case"/"results")]:
        real=read(folder/"real_formal_replications.csv"); quasi=read(folder/"quasi_formal_replications.csv")
        sources += [str((folder/n).relative_to(ROOT)) for n in ("real_formal_replications.csv","quasi_formal_replications.csv")]
        for scenario,g in real.groupby("scenario",sort=False):
            conv=g.converged.map(flag).to_numpy(); n=len(g); truth=float(num(g.A_true).mean())
            for method in ("model","hac14","hac28"):
                cover=g["cover_"+method].map(flag).to_numpy(dtype=float); width=num(g["width_"+method])
                cp=float(cover.mean()); cc=float(cover[conv].mean()) if conv.any() else float("nan")
                err=num(g.A_hat)-num(g.A_true)
                covrows.append({"case":label,"scenario":scenario,"method":method,"A_true":truth,"n_planned":n,
                    "n_converged":int(conv.sum()),"n_failed":int((~conv).sum()),"coverage_planned":cp,
                    "coverage_mcse_planned":math.sqrt(cp*(1-cp)/n),"coverage_converged_only":cc,
                    "coverage_mcse_converged_only":math.sqrt(cc*(1-cc)/int(conv.sum())) if conv.any() else float("nan"),
                    "mean_interval_width_converged":float(width[conv].mean()) if conv.any() else float("nan"),
                    "mean_bias_converged":float(err[conv].mean()) if conv.any() else float("nan"),
                    "mean_phi_hat":float("nan"),"mean_se_quasi_over_empirical_sd":float("nan"),
                    "family":"daily_observed_and_inference"})
            for a,b in (("model","hac14"),("model","hac28"),("hac14","hac28")):
                c1=g["cover_"+a].map(flag).to_numpy(dtype=float); c2=g["cover_"+b].map(flag).to_numpy(dtype=float)
                w1=num(g["width_"+a]); w2=num(g["width_"+b]); ok=conv&np.isfinite(w1)&np.isfinite(w2); wd=w1[ok]-w2[ok]; diff=c1-c2
                realpairs.append({"scenario":scenario,"method_1":a,"method_2":b,"n_planned":n,
                    "mean_paired_coverage_difference":float(diff.mean()),"mcse_paired_coverage_difference":mcse(diff),
                    "n_converged_width_pairs":int(ok.sum()),"mean_paired_width_difference":float(wd.mean()),
                    "mcse_paired_width_difference":mcse(wd),"family":"daily_real_paired","case":label})
        for scenario,g in quasi.groupby("scenario",sort=False):
            conv=g.converged.map(flag).to_numpy(); n=len(g); cover=g.cover_quasi.map(flag).to_numpy(dtype=float)
            width=num(g.width_quasi); err=num(g.A_hat)-num(g.A_true); esd=float(np.std(err[conv],ddof=1)) if conv.sum()>1 else float("nan")
            se=num(g.se_quasi); cp=float(cover.mean())
            cc=float(cover[conv].mean()) if conv.any() else float("nan")
            covrows.append({"case":label,"scenario":scenario,"method":"quasi_poisson_pearson_scale",
                "A_true":float(num(g.A_true).mean()),"n_planned":n,"n_converged":int(conv.sum()),"n_failed":int((~conv).sum()),
                "coverage_planned":cp,"coverage_mcse_planned":math.sqrt(cp*(1-cp)/n),
                "coverage_converged_only":cc,"coverage_mcse_converged_only":math.sqrt(cc*(1-cc)/int(conv.sum())) if conv.any() else float("nan"),
                "mean_interval_width_converged":float(width[conv].mean()) if conv.any() else float("nan"),
                "mean_bias_converged":float(err[conv].mean()) if conv.any() else float("nan"),
                "mean_phi_hat":float(num(g.phi_hat)[conv].mean()) if conv.any() else float("nan"),
                "mean_se_quasi_over_empirical_sd":float(se[conv].mean()/esd) if np.isfinite(esd) and esd>0 else float("nan"),
                "family":"daily_observed_and_inference"})
            for b in ("model","hac14","hac28"):
                c2=g["cover_"+b].map(flag).to_numpy(dtype=float); w2=num(g["width_"+b]); ok=conv&np.isfinite(width)&np.isfinite(w2); wd=width[ok]-w2[ok]; diff=cover-c2
                quasipairs.append({"case":"city" if label=="municipality" else "state","scenario":scenario,
                    "method_1":"quasi","method_2":b,"n_planned":n,"mean_paired_coverage_difference":float(diff.mean()),
                    "mcse_paired_coverage_difference":mcse(diff),"n_converged_width_pairs":int(ok.sum()),
                    "mean_paired_width_difference_quasi_minus_comparator":float(wd.mean()),
                    "mcse_paired_width_difference":mcse(wd),"family":"daily_quasi_paired"})
    return pd.DataFrame(covrows),pd.DataFrame(realpairs),pd.DataFrame(quasipairs),sources

def mvn_sources():
    parts=[]; sources=[]
    for case in ("state","city"):
        path=ROOT/"additional_analyses"/"results"/f"mvn_formal_{case}_replications.csv"
        d=read(path); d["case"]=case; parts.append(d); sources.append(str(path.relative_to(ROOT)))
    return pd.concat(parts,ignore_index=True),sources

def mvn_summary(d):
    rows=[]; d=d.loc[d.covariance_method.isin({"quasi_pearson","hac28"})]
    for keys,g in d.groupby(["case","scenario","covariance_method","interval_method"],sort=False):
        case,sc,cov,method=keys; valid=g.interval_valid.map(flag).to_numpy()
        covered=g.covered.map(flag).to_numpy(dtype=float); n=len(g); cp=float(covered.mean())
        w=num(g.interval_width); vw=w[valid&np.isfinite(w)]; asym=num(g.interval_asymmetry_upper_minus_lower)
        shift=num(g.interval_center_shift_from_A_hat); psd=g.covariance_psd_valid.map(flag).to_numpy()
        rows.append({"case":case,"scenario":sc,"covariance_method":cov,"interval_method":method,"n_planned":n,
            "n_valid_intervals":int(valid.sum()),"n_invalid_intervals_counted_uncovered":int(n-valid.sum()),
            "n_covered_planned":int(covered.sum()),"coverage_planned":cp,
            "coverage_mcse_planned":math.sqrt(cp*(1-cp)/n),"coverage_valid_only":float(covered[valid].mean()) if valid.any() else float("nan"),
            "mean_width_valid_only":float(vw.mean()) if len(vw) else float("nan"),"mcse_mean_width_valid_only":mcse(vw),
            "mean_interval_asymmetry_upper_minus_lower":float(asym[valid&np.isfinite(asym)].mean()) if (valid&np.isfinite(asym)).any() else float("nan"),
            "mcse_mean_interval_asymmetry_valid_only":mcse(asym[valid&np.isfinite(asym)]),
            "mean_interval_center_shift_from_A_hat":float(shift[valid&np.isfinite(shift)].mean()) if (valid&np.isfinite(shift)).any() else float("nan"),
            "n_psd_covariances":int(psd.sum()),"n_nonpsd_covariances":int((~psd).sum())})
    return pd.DataFrame(rows)

def matched(left,right,keys,cov1,cov2,m1,m2):
    l=left.loc[(left.covariance_method==cov1)&(left.interval_method==m1)]
    r=right.loc[(right.covariance_method==cov2)&(right.interval_method==m2)]
    j=l.merge(r,on=keys,suffixes=("_1","_2"),validate="one_to_one")
    if j.empty: raise ValueError(f"no matched rows for {cov1}/{m1} vs {cov2}/{m2}")
    c1=j.covered_1.map(flag).to_numpy(dtype=float); c2=j.covered_2.map(flag).to_numpy(dtype=float); diff=c1-c2
    valid=j.interval_valid_1.map(flag).to_numpy()&j.interval_valid_2.map(flag).to_numpy()
    w1=num(j.interval_width_1); w2=num(j.interval_width_2); ok=valid&np.isfinite(w1)&np.isfinite(w2); wd=w1[ok]-w2[ok]
    return {"n":len(j),"coverage":float(diff.mean()),"coverage_mcse":mcse(diff),"nwidth":int(ok.sum()),
            "width":float(wd.mean()) if len(wd) else float("nan"),"width_mcse":mcse(wd)}

def mvn_pairs(d):
    methodrows=[]; covrows=[]; ids=["case","scenario","replicate"]
    for (case,sc),g in d.groupby(["case","scenario"],sort=False):
        for cov in ("quasi_pearson","hac28"):
            for method in ("normal_wald","linear_gaussian_mc"):
                x=matched(g,g,ids,cov,cov,"exact_nonlinear_mc",method)
                methodrows.append({"case":case,"scenario":sc,"covariance_method":cov,"method_1":"exact_nonlinear_mc",
                    "method_2":method,"n_planned_pairs":x["n"],"mean_paired_coverage_difference_method_1_minus_method_2":x["coverage"],
                    "mcse_paired_coverage_difference":x["coverage_mcse"],"n_jointly_valid_width_pairs":x["nwidth"],
                    "mean_paired_width_difference_method_1_minus_method_2":x["width"],"mcse_paired_width_difference":x["width_mcse"]})
        for method in ("normal_wald","linear_gaussian_mc","exact_nonlinear_mc"):
            x=matched(g,g,ids,"quasi_pearson","hac28",method,method)
            covrows.append({"case":case,"scenario":sc,"interval_method":method,"method_1":"quasi_pearson","method_2":"hac28",
                "n_planned_pairs":x["n"],"mean_paired_coverage_difference_method_1_minus_method_2":x["coverage"],
                "mcse_paired_coverage_difference":x["coverage_mcse"],"n_jointly_valid_width_pairs":x["nwidth"],
                "mean_paired_width_difference_method_1_minus_method_2":x["width"],"mcse_paired_width_difference":x["width_mcse"]})
    covariance_pairs = pd.DataFrame(covrows).rename(columns={
        "mean_paired_coverage_difference_method_1_minus_method_2":"mean_paired_coverage_difference_quasi_minus_hac28",
        "mean_paired_width_difference_method_1_minus_method_2":"mean_paired_width_difference_quasi_minus_hac28"})
    return pd.DataFrame(methodrows),covariance_pairs

def stability_tables():
    pieces=[]; sources=[]
    for case in ("city","state"):
        p=ROOT/"additional_analyses"/"results"/f"mvn_formal_{case}_stability.csv"
        pieces.append(read(p)); sources.append(str(p.relative_to(ROOT)))
    detail=pd.concat(pieces,ignore_index=True); detail["n_draws"]=pd.to_numeric(detail.n_draws,errors="raise").astype(int)
    base=detail.loc[detail.n_draws==2048,["case","scenario","replicate","covariance_method","interval_method","interval_width","covered"]].rename(
        columns={"interval_width":"base_width","covered":"base_covered"})
    t=detail.loc[detail.n_draws.isin([4096,8192])].merge(base,on=["case","scenario","replicate","covariance_method","interval_method"],how="left",validate="many_to_one")
    if t.base_width.isna().any(): raise ValueError("missing matched 2048-draw stability baseline")
    den=np.abs(num(t.base_width)); wc=num(t.width_change_vs_2048)
    lo=np.abs(num(t.lower_change_vs_2048)); hi=np.abs(num(t.upper_change_vs_2048))
    t["relative"]=np.divide(np.abs(wc),den,out=np.full(len(t),np.nan),where=den>0)
    t["endpoint"]=np.divide(np.maximum(lo,hi),den,out=np.full(len(t),np.nan),where=den>0)
    t["flip"]=[flag(a)!=flag(b) for a,b in zip(t.covered,t.base_covered)]
    rows=[]
    for k,g in t.groupby(["case","covariance_method","interval_method","n_draws"],sort=False):
        v=num(g.relative); e=num(g.endpoint)
        rows.append({"case":k[0],"covariance_method":k[1],"interval_method":k[2],"n_draws":int(k[3]),
            "n_intervals":int(g[["scenario","replicate"]].drop_duplicates().shape[0]),
            "mean_absolute_relative_width_change":float(np.nanmean(v)),"median_absolute_relative_width_change":float(np.nanmedian(v)),
            "max_absolute_relative_width_change":float(np.nanmax(v)),"max_endpoint_change_divided_by_width":float(np.nanmax(e)),
            "coverage_flips":int(g.flip.sum())})
    return pd.DataFrame(rows),detail,sources

def write(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False,encoding="utf-8-sig")

def compare(actual,expected,keys,rtol=1e-8,atol=1e-8):
    if not actual.is_file() or not expected.is_file(): return {"ok":False,"reason":"missing actual or baseline"}
    a=pd.read_csv(actual); b=pd.read_csv(expected); missing=[c for c in b if c not in a]
    if missing: return {"ok":False,"reason":f"missing baseline fields: {missing}"}
    if len(a)!=len(b): return {"ok":False,"reason":f"row count differs: rebuilt={len(a)}, baseline={len(b)}"}
    use=[k for k in keys if k in a and k in b]
    try: a=a.set_index(use).sort_index(); b=b.set_index(use).sort_index()
    except Exception as e: return {"ok":False,"reason":f"could not align by {use}: {e}"}
    if not a.index.is_unique or not b.index.is_unique: return {"ok":False,"reason":"comparison keys are not unique"}
    if not a.index.equals(b.index): return {"ok":False,"reason":"row keys differ"}
    diffs=[]
    for c in b.columns:
        x,y=a[c],b[c]; xn=pd.to_numeric(x,errors="coerce"); yn=pd.to_numeric(y,errors="coerce")
        if (xn.notna()|x.isna()).all() and (yn.notna()|y.isna()).all():
            if not np.allclose(xn.to_numpy(float),yn.to_numpy(float),rtol=rtol,atol=atol,equal_nan=True):
                z=np.abs(xn.to_numpy(float)-yn.to_numpy(float)); z=z[np.isfinite(z)]
                diffs.append({"field":c,"max_abs_diff":float(z.max()) if len(z) else None})
        elif not np.array_equal(x.fillna("").astype(str).to_numpy(),y.fillna("").astype(str).to_numpy()):
            diffs.append({"field":c,"reason":"non-numeric values differ"})
    return {"ok":not diffs,"rows":len(a),"compared_fields":len(b.columns),"differences":diffs[:8]}

def build(output,check):
    output.mkdir(parents=True,exist_ok=True)
    s1,s3,fixedsrc=fixed_tables()
    s4a,s4b,s4c,dailysrc=daily_tables()
    mvn,mvnsrc=mvn_sources(); s10=mvn_summary(mvn); s11,s11b=mvn_pairs(mvn)
    s11c,s11d,stabilitysrc=stability_tables()
    rebuilt={"Table_S1_fixed_method_summaries.csv":s1,"Table_S3_fixed_paired_differences.csv":s3,
        "Table_S4a_daily_coverage.csv":s4a,"Table_S4b_real_paired_differences.csv":s4b,
        "Table_S4c_quasi_paired_differences.csv":s4c,"Table_S10_mvn_summary.csv":s10,
        "Table_S11_mvn_method_paired.csv":s11,"Table_S11b_covariance_paired.csv":s11b,
        "Table_S11c_stability_summary.csv":s11c,"Table_S11d_stability_detailed.csv":s11d}
    classes={}
    for name,frame in rebuilt.items():
        write(frame,output/name)
        classes[name]={"level":"replication_level_detail_assembly" if name.endswith("d_stability_detailed.csv") else "replication_level_resummary","rows":len(frame)}
    for name in COPY_TABLES:
        src=ROOT/"paper"/"tables"/name
        if not src.is_file(): raise FileNotFoundError(f"required copied table absent: {src}")
        shutil.copyfile(src,output/name)
        classes[name]={"level":"copied_scientific_metadata_or_deterministic_analysis","rows":len(pd.read_csv(src)),
                       "source":str(src.relative_to(ROOT)),"note":"Copied from paper/tables; not described as recomputed."}
    main={"Table_1_fixed_design.csv":pd.read_csv(output/"Table_S1_fixed_method_summaries.csv"),
          "Table_2_daily_coverage.csv":pd.read_csv(output/"Table_S4a_daily_coverage.csv"),
          "Table_3_mvn_intervals.csv":pd.read_csv(output/"Table_S10_mvn_summary.csv")}
    bounds=pd.read_csv(output/"Table_S12_allocation_bounds.csv")
    main["Table_4_monthly_bound_delta_0.05.csv"]=bounds.loc[np.isclose(pd.to_numeric(bounds.delta,errors="coerce"),.05)].copy()
    for name,frame in main.items():
        write(frame,output/name)
        classes[name]={"level":"selection_from_rebuilt_supplementary_table" if name!="Table_4_monthly_bound_delta_0.05.csv" else "selection_from_copied_deterministic_analysis",
                       "rows":len(frame),"note":"Selection only; see source table provenance."}
    checks={}
    if check:
        for name,keys in CHECK_KEYS.items(): checks[name]=compare(output/name,ROOT/"paper"/"tables"/name,keys)
        for mn,sn,keys in [("Table_1_fixed_design.csv","Table_S1_fixed_method_summaries.csv",CHECK_KEYS["Table_S1_fixed_method_summaries.csv"]),
                           ("Table_2_daily_coverage.csv","Table_S4a_daily_coverage.csv",CHECK_KEYS["Table_S4a_daily_coverage.csv"]),
                           ("Table_3_mvn_intervals.csv","Table_S10_mvn_summary.csv",CHECK_KEYS["Table_S10_mvn_summary.csv"])]:
            checks[mn]=compare(output/mn,output/sn,keys)
        checks["Table_4_monthly_bound_delta_0.05.csv"]={"ok":len(main["Table_4_monthly_bound_delta_0.05.csv"])==int(np.isclose(pd.to_numeric(bounds.delta,errors="coerce"),.05).sum()),
            "rows":len(main["Table_4_monthly_bound_delta_0.05.csv"]),"compared_fields":len(bounds.columns),"differences":[]}
    report={"generated_utc":datetime.now(timezone.utc).isoformat(),"command":"python scripts/build_tables.py --output outputs/tables"+(" --check" if check else ""),
        "python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":__import__("scipy").__version__,
        "inputs":{"fixed_and_heatday_panels":fixedsrc,"daily_replications":dailysrc,"mvn_replications":mvnsrc,"mvn_stability_replications":stabilitysrc},
        "tables":classes,"checks":checks,"check_tolerance":{"rtol":1e-8,"atol":1e-8},
        "pairing_rule":"MVN pairs match case, scenario, replicate, covariance_method and interval_method; source coefficient matching flags and exact covariance-method identifiers are preserved."}
    report["check_status"]=("passed" if all(v.get("ok",False) for v in checks.values()) else "failed") if check else "not_requested"
    (output/"rebuild_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=str)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"check_status":report["check_status"],"tables":len(classes),"checks":checks},indent=2,ensure_ascii=False,default=str))
    return 1 if check and report["check_status"]!="passed" else 0

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output",type=Path,default=Path("outputs/tables"))
    p.add_argument("--check",action="store_true",help="compare rebuilt CSVs with paper/tables using numeric tolerance")
    a=p.parse_args(); output=a.output if a.output.is_absolute() else ROOT/a.output
    return build(output,a.check)

if __name__=="__main__": raise SystemExit(main())

