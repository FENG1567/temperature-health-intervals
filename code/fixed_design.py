"""Fixed-design outcome-resampling study for attribution intervals."""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, csv, hashlib, importlib.util, json, math, platform, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys as _repro_sys
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'reproduce.py').is_file())
_repro_sys.path.insert(0, str(_REPRO_ROOT / 'code'))
from repro_paths import source_root, run_root, output_path
from typing import Any
import numpy as np
import scipy
from scipy.stats import t as student_t

ROOT = source_root()
CONFIG = ROOT / "inputs" / "fixed_config.json"
SOURCE_FILE = ROOT / "inputs" / "code_snapshot" / "core_sim.py"
_spec = importlib.util.spec_from_file_location("pinned_core_sim", SOURCE_FILE)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Pinned source not found: {SOURCE_FILE}")
source = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = source
_spec.loader.exec_module(source)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8-sig") as f: cfg = json.load(f)
    got = sha(SOURCE_FILE)
    if got.lower() != cfg["source"]["snapshot_sha256"].lower():
        raise RuntimeError(f"pinned source hash mismatch: {got}")
    return cfg

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader()
        for r in rows:
            encoded={}
            for k,v in r.items():
                if isinstance(v,(float,np.floating)):
                    if np.isnan(v): v=""
                    elif np.isposinf(v): v="inf"
                    elif np.isneginf(v): v="-inf"
                encoded[k]=v
            w.writerow(encoded)
    tmp.replace(path)

def qfinite(scores: np.ndarray, alpha: float) -> float:
    s=np.asarray(scores,float); s=s[np.isfinite(s)]; n=len(s)
    if not n: return float("nan")
    k=max(int(math.ceil((n+1)*(1-alpha))),1)
    if k>n: return float("inf")
    return float(np.sort(s)[k-1])

def make_design(cfg: dict[str,Any], sc: dict[str,Any], seed: int) -> dict[str,Any]:
    d=cfg["design"]
    if int(d["n_c"])!=1: raise ValueError("protocol requires n_c=1")
    c=source.Cfg(n_c=1,R=int(d["R"]),T=int(d["T"]),theta=float(d["theta"]),
                 misspec=sc["kind"]=="source_misspec",bias=0.0,common_bias=0.0,label=sc["name"])
    dat=source.simulate(c,int(seed)); R,T=c.R,c.T
    knots=np.quantile(dat["x_lag"][:,:,:,0].reshape(-1),[.15,.5,.85])
    x=dat["x_lag"][0].reshape(-1,3); xcf=dat["xcf_lag"][0].reshape(-1,3)
    region=np.repeat(np.arange(R),T); month=np.tile(np.arange(T)%12,R)
    trend=np.tile(np.arange(T)/12.,R); year=np.tile(np.arange(T)//12,R)
    off=np.log(np.repeat(dat["pop"][0],T))
    X=source.build_X(x,region,month,trend,knots,R)
    Xcf=source.cf_X(X,xcf,knots,shift=0.)
    mu=np.asarray(dat["mu"][0],float).reshape(-1).copy()
    mucf=np.asarray(dat["mucf"][0],float).reshape(-1).copy()
    beta=np.full(X.shape[1],np.nan); proj_rmse=float("nan")
    pc_fit_ok=None; pc_beta_max_abs_error=float("nan"); pc_mu_max_abs_error=float("nan")
    if sc["kind"]=="exact_model":
        lm=np.log(np.maximum(mu,1e-12))-off
        beta=np.linalg.lstsq(X,lm,rcond=None)[0]
        proj_rmse=float(np.sqrt(np.mean((lm-X@beta)**2)))
        mu=np.exp(np.clip(X@beta+off,-30,30)); mucf=np.exp(np.clip(Xcf@beta+off,-30,30))
        pc=fit_tw(mu,X,off,region,year,cfg)
        pc_beta_max_abs_error=float(np.max(np.abs(pc["beta"]-beta)))
        pc_mu=np.exp(np.clip(X@pc["beta"]+off,-30,30))
        pc_mu_max_abs_error=float(np.max(np.abs(pc_mu-mu)))
        pc_fit_ok=bool(pc["conv"] and pc["rank"]==X.shape[1]
          and np.isfinite(pc_beta_max_abs_error) and pc_beta_max_abs_error<1e-6
          and np.isfinite(pc_mu_max_abs_error) and pc_mu_max_abs_error<1e-6)
        if not pc_fit_ok:
            raise RuntimeError(f"exact-model positive-control fit failed: converged={pc['conv']}, rank={pc['rank']}/{X.shape[1]}, beta_error={pc_beta_max_abs_error}, mu_error={pc_mu_max_abs_error}")
    truth=float(np.sum(mu-mucf))
    z={"scenario":sc["name"],"scenario_id":int(sc["id"]),"design_seed":int(seed),"cfg":c,
       "X":X,"Xcf":Xcf,"off":off,"region":region,"year":year,"mu":mu,"mucf":mucf,
       "truth":truth,"theta":float(d["theta"]),"beta":beta,"projection_rmse":proj_rmse,
       "positive_control_fit_ok":pc_fit_ok,"positive_control_beta_max_abs_error":pc_beta_max_abs_error,
       "positive_control_mu_max_abs_error":pc_mu_max_abs_error,
       "knots":knots,"pop":dat["pop"][0],"xlag":dat["x_lag"][0],"xcflag":dat["xcf_lag"][0]}
    out=run_root()/"results"/"fixed_designs"/f"fixed_design_{sc['name']}_seed{seed}.npz"
    out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,X=X,Xcf=Xcf,offset=off,region=region,year=year,mu=mu,mucf=mucf,
          beta_fixed=beta,projection_rmse=proj_rmse,knots=knots,pop=dat["pop"][0],x_lag=dat["x_lag"][0],
          xcf_lag=dat["xcf_lag"][0],A_true=truth,theta=c.theta,
          metadata_json=np.asarray(json.dumps({"scenario":sc["name"],"seed":seed,"R":R,"T":T,
            "source_sha256":sha(SOURCE_FILE),"truth":truth,"positive_control_fit_ok":pc_fit_ok,
            "positive_control_beta_max_abs_error":pc_beta_max_abs_error,
            "positive_control_mu_max_abs_error":pc_mu_max_abs_error},ensure_ascii=False)))
    tmp.replace(out); z["design_path"]=str(out); z["design_sha256"]=sha(out)
    return z

def meat(Xu: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray,int]:
    _,inv=np.unique(np.asarray(groups),return_inverse=True)
    agg=np.zeros((int(inv.max())+1,Xu.shape[1]))
    np.add.at(agg,inv,Xu)
    return agg.T@agg,len(agg)

def fit_tw(y: np.ndarray, X: np.ndarray, off: np.ndarray, region: np.ndarray, year: np.ndarray, cfg: dict[str,Any]) -> dict[str,Any]:
    """Source IRLS with corrected region x year grouped-intersection CR1."""
    y=np.asarray(y,float); b=np.zeros(X.shape[1]); n,p=X.shape
    conv=False; step=float("inf"); used_lstsq=False
    for it in range(int(cfg["fit"]["max_iterations"])):
        eta=np.clip(X@b+off,-30,30); mu=np.exp(eta); w=np.maximum(mu,1e-8)
        z=eta-off+(y-mu)/w; A=X.T@(w[:,None]*X); rhs=X.T@(w*z)
        try: bn=np.linalg.solve(A,rhs)
        except np.linalg.LinAlgError: bn=np.linalg.lstsq(A,rhs,rcond=None)[0]; used_lstsq=True
        step=float(np.max(np.abs(bn-b))); b=bn
        if step<float(cfg["fit"]["tolerance_max_abs_step"]): conv=True; break
    mu=np.exp(np.clip(X@b+off,-30,30)); w=np.maximum(mu,1e-8); A=X.T@(w[:,None]*X)
    rank=int(np.linalg.matrix_rank(A)); cond=float(np.linalg.cond(A))
    try: Ai=np.linalg.inv(A); used_pinv=False
    except np.linalg.LinAlgError: Ai=np.linalg.pinv(A); used_pinv=True
    Xu=X*(y-mu)[:,None]
    Mr,nr=meat(Xu,region); My,ny=meat(Xu,year)
    cross=region.astype(np.int64)*(int(year.max())+1)+year.astype(np.int64)
    Mx,nx=meat(Xu,cross); dfc=(n-1)/max(n-p,1)
    corr=lambda g:(g/max(g-1,1))*dfc
    G=min(nr,ny); common_corr=(G/max(G-1,1))*dfc
    # Component-specific correction is the primary two-way CR1 variant.
    V_component=Ai@(corr(nr)*Mr+corr(ny)*My-corr(nx)*Mx)@Ai
    V_component=(V_component+V_component.T)/2
    # Use one common min-cluster correction and the same PSD clip in both matrices
    # to isolate the intersection-meat change from the pinned-source formula.
    V_common_raw=Ai@(Mr+My-Mx)@Ai*common_corr; V_common_raw=(V_common_raw+V_common_raw.T)/2
    V_legacy_raw=Ai@(Mr+My-Xu.T@Xu)@Ai*common_corr; V_legacy_raw=(V_legacy_raw+V_legacy_raw.T)/2
    def psd_clip(V):
        ev,U=np.linalg.eigh(V); mineig=float(ev.min()); clipped=int(np.sum(ev<0))
        return (U*np.maximum(ev,0))@U.T,mineig,clipped
    V_common,mineig_common,clip_common=psd_clip(V_common_raw)
    V_legacy,mineig_legacy,clip_legacy=psd_clip(V_legacy_raw)
    # Contextual benchmarks for independent-observation HC1 and one-way region CR1.
    V_hc1=Ai@(Xu.T@Xu)@Ai*(n/max(n-p,1)); V_hc1=(V_hc1+V_hc1.T)/2
    region_corr=(nr/max(nr-1,1))*dfc
    V_region=Ai@(region_corr*Mr)@Ai; V_region=(V_region+V_region.T)/2
    return {"beta":b,"V_component":V_component,"V_common":V_common,"V_legacy":V_legacy,
      "V_hc1":V_hc1,"V_region":V_region,"mu":mu,"iter":it+1,"conv":conv,"step":step,
      "rank":rank,"cond":cond,"mineig_component":float(np.linalg.eigvalsh(V_component).min()),
      "mineig_common_raw":mineig_common,"mineig_legacy_raw":mineig_legacy,
      "clip_common":clip_common,"clip_legacy":clip_legacy,"nr":nr,"ny":ny,"nx":nx,
      "lstsq":used_lstsq,"pinv":used_pinv,"df":max(min(nr,ny)-1,1),"df_region":max(nr-1,1),
      "df_hc1":max(n-p,1)}

def fit_rep(y: np.ndarray, D: dict[str,Any], cfg: dict[str,Any]) -> dict[str,Any]:
    try:
        f=fit_tw(y.reshape(-1),D["X"],D["off"],D["region"],D["year"],cfg)
        b=f["beta"]; m=np.exp(np.clip(D["X"]@b+D["off"],-30,30)); mc=np.exp(np.clip(D["Xcf"]@b+D["off"],-30,30))
        ah=float(np.sum(m-mc)); g=D["X"].T@m-D["Xcf"].T@mc
        variances={k:float(g@f[vkey]@g) for k,vkey in {
          "corrected":"V_component","common":"V_common","legacy":"V_legacy",
          "hc1":"V_hc1","region":"V_region"}.items()}
        se={k:math.sqrt(v) if np.isfinite(v) and v>0 else float("nan") for k,v in variances.items()}
        alpha=float(cfg["sampling"]["alpha"])
        df=f["df"]; te=float(student_t.ppf(1-alpha/2,df)); tq=float(source.tq(df))
        te_region=float(student_t.ppf(1-alpha/2,f["df_region"]))
        te_hc1=float(student_t.ppf(1-alpha/2,f["df_hc1"]))
        flags=[]
        if not f["conv"]: flags.append("not_converged")
        if f["rank"]<D["X"].shape[1]: flags.append("rank_deficient")
        if f["cond"]>float(cfg["fit"]["condition_warning_threshold"]): flags.append("condition_warning")
        for name,value in se.items():
            if not np.isfinite(value): flags.append("nonpositive_"+name+"_variance")
        coefficient_ok=bool(f["conv"] and f["rank"]==D["X"].shape[1]
          and np.all(np.isfinite(b)) and np.isfinite(ah) and np.all(np.isfinite(m)) and np.all(np.isfinite(mc)))
        return {"A_true":D["truth"],"A_hat":ah,"error":ah-D["truth"],
          "h_corrected":se["corrected"],"h_common_cr1":se["common"],"h_legacy":se["legacy"],
          "h_hc1":se["hc1"],"h_region_cr1":se["region"],
          "variance_corrected":variances["corrected"],"variance_common_cr1":variances["common"],
          "variance_legacy":variances["legacy"],"variance_hc1":variances["hc1"],
          "variance_region_cr1":variances["region"],
          "corrected_variance_ok":bool(np.isfinite(se["corrected"])),
          "common_cr1_variance_ok":bool(np.isfinite(se["common"])),
          "legacy_variance_ok":bool(np.isfinite(se["legacy"])),
          "hc1_variance_ok":bool(np.isfinite(se["hc1"])),"region_cr1_variance_ok":bool(np.isfinite(se["region"])),
          "df":df,"df_region":f["df_region"],"df_hc1":f["df_hc1"],"tcrit_exact":te,
          "tcrit_region":te_region,"tcrit_hc1":te_hc1,"tcrit_source_tq":tq,
          "fit_ok":coefficient_ok,"coefficient_fit_ok":coefficient_ok,"fit_converged":f["conv"],
          "fit_iterations":f["iter"],"max_abs_step":f["step"],"condition_number":f["cond"],
          "cov_min_eigenvalue":f["mineig_component"],"common_cov_raw_min_eigenvalue":f["mineig_common_raw"],
          "legacy_cov_raw_min_eigenvalue":f["mineig_legacy_raw"],
          "common_eigenvalues_clipped":f["clip_common"],"legacy_eigenvalues_clipped":f["clip_legacy"],
          "n_region_clusters":f["nr"],"n_year_clusters":f["ny"],"n_intersections":f["nx"],
          "used_lstsq":f["lstsq"],"used_pinv":f["pinv"],
          "fit_flags":";".join(flags)}
    except Exception as e:
        out={"A_true":D["truth"],"A_hat":float("nan"),"error":float("nan"),"fit_ok":False,
          "coefficient_fit_ok":False,"fit_converged":False,"fit_iterations":0,"max_abs_step":float("nan"),
          "condition_number":float("nan"),"cov_min_eigenvalue":float("nan"),"common_cov_raw_min_eigenvalue":float("nan"),
          "legacy_cov_raw_min_eigenvalue":float("nan"),"common_eigenvalues_clipped":0,"legacy_eigenvalues_clipped":0,
          "used_lstsq":False,"used_pinv":False,"fit_flags":f"fit_exception:{type(e).__name__}:{e}"}
        for key in ("h_corrected","h_common_cr1","h_legacy","h_hc1","h_region_cr1",
          "variance_corrected","variance_common_cr1","variance_legacy","variance_hc1","variance_region_cr1",
          "df","df_region","df_hc1","tcrit_exact","tcrit_region","tcrit_hc1","tcrit_source_tq"):
            out[key]=float("nan")
        for key in ("corrected_variance_ok","common_cr1_variance_ok","legacy_variance_ok","hc1_variance_ok","region_cr1_variance_ok"):
            out[key]=False
        return out

def draw_y(D: dict[str,Any], cfg: dict[str,Any], scid: int, mode: str, stage: str, rep: int) -> np.ndarray:
    sid=int(cfg["sampling"]["stage_ids"][stage]); mode_id=int(cfg["sampling"]["mode_ids"][mode])
    ss=np.random.SeedSequence([int(cfg["sampling"]["base_seed"]),mode_id,scid,D["design_seed"],sid,rep])
    rng=np.random.default_rng(ss); th=D["theta"]; mu=D["mu"].reshape(1,-1)
    return rng.negative_binomial(th,th/(th+mu)).reshape(-1)

def interval(center: float,h: float,q: float,truth: float) -> tuple[float,float,int,float]:
    if not all(np.isfinite(x) for x in (center,h,truth)) or math.isnan(q) or h<=0 or q<0: return (float("nan"),float("nan"),0,float("nan"))
    lo=center-q*h; hi=center+q*h
    return lo,hi,int(lo<=truth<=hi),hi-lo

def full_grid_attribution(beta: np.ndarray, D: dict[str,Any]) -> tuple[float,float]:
    m=np.exp(np.clip(D["X"]@beta+D["off"],-30,30))
    mc=np.exp(np.clip(D["Xcf"]@beta+D["off"],-30,30))
    a=float(np.sum(m-mc)); g=D["X"].T@m-D["Xcf"].T@mc
    return a,g

def yearhalf_q_diagnostic(y: np.ndarray, D: dict[str,Any], cfg: dict[str,Any], scid: int,
                          mode: str, rep: int) -> dict[str,Any]:
    """Split calendar years, fit all coefficients, and predict on the full fixed grid."""
    spec=cfg["year_split_diagnostic"]; years=np.unique(D["year"])
    n_keep=int(spec.get("years_per_half",len(years)//2)); ksplit=int(spec["ksplit"])
    if len(years)!=10 or n_keep!=5:
        raise ValueError(f"year-half diagnostic requires ten years and five per half; got {len(years)} and {n_keep}")
    mode_id=int(cfg["sampling"]["mode_ids"][mode])
    ss=np.random.SeedSequence([int(cfg["sampling"]["base_seed"]),mode_id,scid,D["design_seed"],
      int(cfg["sampling"]["stage_ids"]["test"]),rep,int(spec["seed_offset"])])
    rng=np.random.default_rng(ss); zs=[]; failed=0
    for _ in range(ksplit):
        keep=rng.choice(years,size=n_keep,replace=False); mask1=np.isin(D["year"],keep); mask2=~mask1
        split=[]
        for mask in (mask1,mask2):
            f=fit_tw(y[mask],D["X"][mask],D["off"][mask],D["region"][mask],D["year"][mask],cfg)
            a,g=full_grid_attribution(f["beta"],D); var=float(g@f["V_component"]@g)
            ok=bool(f["conv"] and f["rank"]==D["X"].shape[1] and np.isfinite(a) and np.isfinite(var) and var>0)
            split.append((a,var,ok))
        if all(z[2] for z in split):
            denom=math.sqrt(split[0][1]+split[1][1]); score=abs(split[0][0]-split[1][0])/denom
            if np.isfinite(score): zs.append(float(score)); continue
        failed+=1
    q=qfinite(np.asarray(zs,float),float(cfg["sampling"]["alpha"]))
    return {"q":q,"n_split_requested":ksplit,"n_split_valid":len(zs),"n_split_failed":failed,
      "split_scores":zs}

def summarize(rows: list[dict[str,Any]], method: str, sc: str, seed: int,
              eval_rows: list[dict[str,Any]]|None=None) -> dict[str,Any]:
    rr=rows if eval_rows is None else eval_rows
    covers=np.array([r.get("cover__"+method,0) for r in rr],float)
    valid=np.array([r.get("interval_valid__"+method,False) for r in rr],bool)
    fits=np.array([r.get("fit_ok",False) for r in rr],bool)
    widths=np.array([r.get("width__"+method,np.nan) for r in rr],float)
    valid_widths=widths[valid & ~np.isnan(widths)]
    finite_widths=valid_widths[np.isfinite(valid_widths)]; n=len(rr)
    cov=float(covers.mean()) if n else float("nan")
    cov_fit=float(covers[fits].mean()) if fits.any() else float("nan")
    cov_valid=float(covers[valid].mean()) if valid.any() else float("nan")
    err=np.array([r["error"] for r in rr if np.isfinite(r["error"])],float)
    inf_widths=int(np.isinf(valid_widths).sum())
    mean_width=float("inf") if inf_widths else (float(finite_widths.mean()) if len(finite_widths) else float("nan"))
    if inf_widths: width_mcse=float("inf")
    else: width_mcse=float(finite_widths.std(ddof=1)/math.sqrt(len(finite_widths))) if len(finite_widths)>1 else float("nan")
    return {"scenario":sc,"design_seed":seed,"method":method,"n_planned":n,"coverage":cov,
      "coverage_mcse_1se":math.sqrt(cov*(1-cov)/n) if n and np.isfinite(cov) else float("nan"),
      "coverage_90pct_mc_halfwidth":1.645*math.sqrt(cov*(1-cov)/n) if n and np.isfinite(cov) else float("nan"),
      "n_fit_success":int(fits.sum()),"coverage_given_fit_success":cov_fit,
      "n_interval_valid":int(valid.sum()),"coverage_given_valid_interval":cov_valid,
      "n_width_valid":len(valid_widths),"n_width_finite":len(finite_widths),"n_width_infinite":inf_widths,
      "mean_width":mean_width,"mean_finite_width":float(finite_widths.mean()) if len(finite_widths) else float("nan"),
      "width_mcse_1se":width_mcse,
      "mean_error":float(err.mean()) if len(err) else float("nan"),"sd_error":float(err.std(ddof=1)) if len(err)>1 else float("nan"),
      "n_fit_fail":sum(not r.get("fit_ok",False) for r in rr),"n_nonconverged":sum(not r.get("fit_converged",False) for r in rr),
      "n_method_interval_invalid_after_fit":sum(r.get("fit_ok",False) and not r.get("interval_valid__"+method,False) for r in rr),
      "n_condition_warning":sum("condition_warning" in r.get("fit_flags","") for r in rr)}

def run_panel(task: tuple[str,int,str]) -> dict[str,Any]:
    scname,seed,mode=task; cfg=read_config(); sc=next(s for s in cfg["design"]["scenarios"] if s["name"]==scname)
    D=make_design(cfg,sc,seed); counts=cfg["sampling"][f"{mode}_reps_per_panel"]; stages={}
    mode_id=int(cfg["sampling"]["mode_ids"][mode])
    for stage in ("bias_train","scale_calibration","test"):
        rows=[]
        for i in range(int(counts[stage])):
            y=draw_y(D,cfg,int(sc["id"]),mode,stage,i); r=fit_rep(y,D,cfg)
            r.update({"scenario":scname,"scenario_id":int(sc["id"]),"design_seed":seed,
              "mode":mode,"mode_id":mode_id,"stage":stage,"replicate":i})
            rows.append(r)
        stages[stage]=rows
    tr,cal,te=stages["bias_train"],stages["scale_calibration"],stages["test"]
    errs=np.array([r["error"] for r in tr if r["fit_ok"] and np.isfinite(r["error"])])
    bias=float(errs.mean()) if len(errs) else float("nan")
    valid_cal=[r for r in cal if r["fit_ok"] and np.isfinite(r["error"])
      and np.isfinite(r["h_corrected"]) and r["h_corrected"]>0]
    scores=np.array([abs(r["error"])/r["h_corrected"] for r in valid_cal],float)
    bscores=np.array([abs(r["error"]-bias)/r["h_corrected"] for r in valid_cal if np.isfinite(bias)],float)
    alpha=float(cfg["sampling"]["alpha"]); qraw=qfinite(scores,alpha); qb=qfinite(bscores,alpha)
    shift=float(cfg["shared_additive_shift"])
    bias_common=float(bias+shift) if np.isfinite(bias) else float("nan")
    scores_common=np.array([abs(r["error"]+shift)/r["h_corrected"] for r in valid_cal],float)
    bscores_common=np.array([abs(r["error"]+shift-bias_common)/r["h_corrected"] for r in valid_cal if np.isfinite(bias_common)],float)
    qraw_common=qfinite(scores_common,alpha); qb_common=qfinite(bscores_common,alpha)
    centered_score_change=max([abs((abs(r["error"]+shift-bias_common)-abs(r["error"]-bias))/r["h_corrected"])
      for r in valid_cal if np.isfinite(bias) and np.isfinite(bias_common)] or [0.])
    year_n=min(int(cfg["year_split_diagnostic"][f"{mode}_test_reps_per_panel"]),len(te))
    year_n=int(year_n) if cfg["year_split_diagnostic"].get("enabled",False) else 0
    for i,r in enumerate(te):
        a=r["A_hat"]; truth=r["A_true"]; r["A_hat_common_bias"]=a+shift if np.isfinite(a) else float("nan")
        r["error_common_bias"]=r["error"]+shift if np.isfinite(r["error"]) else float("nan")
        r["shared_additive_shift"]=shift; year=None
        if i<year_n and r["fit_ok"]:
            y=draw_y(D,cfg,int(sc["id"]),mode,"test",i)
            year=yearhalf_q_diagnostic(y,D,cfg,int(sc["id"]),mode,i)
            r["yearhalf_q"]=year["q"];r["yearhalf_splits_requested"]=year["n_split_requested"]
            r["yearhalf_splits_valid"]=year["n_split_valid"];r["yearhalf_splits_failed"]=year["n_split_failed"]
        else:
            r["yearhalf_q"]=float("nan");r["yearhalf_splits_requested"]=0
            r["yearhalf_splits_valid"]=0;r["yearhalf_splits_failed"]=0
        specs={
          "delta_corrected_cr1_tw":(a,r["h_corrected"],r["tcrit_exact"]),
          "delta_corrected_common_cr1_tw":(a,r["h_common_cr1"],r["tcrit_exact"]),
          "delta_legacy_covariance":(a,r["h_legacy"],r["tcrit_exact"]),
          "delta_pinned_source":(a,r["h_legacy"],r["tcrit_source_tq"]),
          "delta_region_cr1_context":(a,r["h_region_cr1"],r["tcrit_region"]),
          "delta_independent_HC1_positive_control":(a,r["h_hc1"],r["tcrit_hc1"]),
          "independent_scale_calibrated":(a,r["h_corrected"],qraw),
          "truth_trained_bias_centered_split_calibrated":(a-bias,r["h_corrected"],qb),
          "estimator_only_common_bias":(a+shift,r["h_corrected"],r["tcrit_exact"]),
          "estimator_only_common_bias_bias_centered":(a+shift-bias_common,r["h_corrected"],qb_common),
        }
        if i<year_n:
            specs["year_half_split_q_diagnostic"]=(a,r["h_corrected"],r["yearhalf_q"])
        for method,(center,hh,q) in specs.items():
            lo,hi,cov,w=interval(center,hh,q,truth) if r["fit_ok"] else (float("nan"),float("nan"),0,float("nan"))
            valid_interval=not (math.isnan(lo) or math.isnan(hi) or math.isnan(w))
            r["lo__"+method]=lo;r["hi__"+method]=hi;r["cover__"+method]=cov
            r["width__"+method]=w;r["interval_valid__"+method]=valid_interval
    # Algebra check: translating both estimate and truth shifts endpoints, errors and widths equivariantly.
    equiv_error=max([abs(((r["A_hat"]+shift)-(r["A_true"]+shift))-r["error"]) for r in te if np.isfinite(r["error"])] or [0.])
    equiv_width=max([abs((r["hi__delta_corrected_cr1_tw"]+shift)-(r["lo__delta_corrected_cr1_tw"]+shift)-r["width__delta_corrected_cr1_tw"])
      for r in te if r.get("interval_valid__delta_corrected_cr1_tw",False) and np.isfinite(r["width__delta_corrected_cr1_tw"])] or [0.])
    common_centered_maxdiff=max([abs((r["A_hat"]+shift-bias_common)-(r["A_hat"]-bias))
      for r in te if np.isfinite(r["A_hat"]) and np.isfinite(bias) and np.isfinite(bias_common)] or [0.])
    methods=["delta_corrected_cr1_tw","delta_corrected_common_cr1_tw","delta_legacy_covariance",
      "delta_pinned_source","delta_region_cr1_context","delta_independent_HC1_positive_control",
      "independent_scale_calibrated","truth_trained_bias_centered_split_calibrated",
      "estimator_only_common_bias","estimator_only_common_bias_bias_centered","year_half_split_q_diagnostic"]
    summaries=[]
    for method in methods:
        eval_rows=te[:year_n] if method=="year_half_split_q_diagnostic" else te
        summaries.append(summarize(te,method,scname,seed,eval_rows=eval_rows))
    pairs=[]; ref="delta_corrected_cr1_tw"
    for method in methods:
        eval_rows=te[:year_n] if method=="year_half_split_q_diagnostic" else te
        a=np.array([r.get("cover__"+method,0) for r in eval_rows],float)
        b=np.array([r.get("cover__"+ref,0) for r in eval_rows],float); d=a-b
        wa=np.array([r.get("width__"+method,np.nan) for r in eval_rows],float)
        wb=np.array([r.get("width__"+ref,np.nan) for r in eval_rows],float)
        ok=np.isfinite(wa)&np.isfinite(wb); wd=wa[ok]-wb[ok]
        pairs.append({"scenario":scname,"design_seed":seed,"method":method,"reference":ref,"n_planned":len(eval_rows),
          "coverage_diff":float(d.mean()) if len(d) else float("nan"),
          "coverage_diff_mcse_paired_1se":float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else float("nan"),
          "n_width_pairs":int(ok.sum()),"n_width_infinite_method":int(np.isinf(wa).sum()),
          "n_width_infinite_reference":int(np.isinf(wb).sum()),
          "mean_width_diff":float(wd.mean()) if len(wd) else float("nan"),
          "width_diff_mcse_paired_1se":float(wd.std(ddof=1)/math.sqrt(len(wd))) if len(wd)>1 else float("nan")})
    allrows=tr+cal+te
    for r in allrows:
        e=r.get("error",float("nan"));h=r.get("h_corrected",float("nan"))
        good=r.get("fit_ok",False) and np.isfinite(e) and np.isfinite(h) and h>0
        r["error_common_bias"]=e+shift if np.isfinite(e) else float("nan")
        r["A_hat_common_bias"]=r["A_hat"]+shift if np.isfinite(r["A_hat"]) else float("nan")
        r["score_raw"]=abs(e)/h if good else float("nan")
        r["score_bias_centered"]=abs(e-bias)/h if good and np.isfinite(bias) else float("nan")
        r["score_common_bias_raw"]=abs(e+shift)/h if good else float("nan")
        r["score_common_bias_centered"]=abs(e+shift-bias_common)/h if good and np.isfinite(bias_common) else float("nan")
        r["bias_hat_train"]=bias;r["bias_hat_train_common_bias"]=bias_common
        r["q_raw_panel"]=qraw;r["q_bias_centered_panel"]=qb
        r["q_raw_common_bias_panel"]=qraw_common;r["q_bias_centered_common_bias_panel"]=qb_common
        r["mode_id"]=mode_id
    panel={"mode":mode,"mode_id":mode_id,"scenario":scname,"scenario_id":int(sc["id"]),
      "design_seed":seed,"A_true":D["truth"],"design_file":str(D["design_path"]),
      "design_sha256":D["design_sha256"],"positive_control_projection_rmse":D["projection_rmse"],
      "positive_control_fit_ok":D["positive_control_fit_ok"],
      "positive_control_beta_max_abs_error":D["positive_control_beta_max_abs_error"],
      "positive_control_mu_max_abs_error":D["positive_control_mu_max_abs_error"],
      "bias_hat_train":bias,"bias_hat_train_common_bias":bias_common,
      "n_bias_train_valid":len(errs),"n_scale_scores":len(scores),"n_bias_scores":len(bscores),
      "q_raw":qraw,"q_bias_centered":qb,"q_raw_common_bias":qraw_common,
      "q_bias_centered_common_bias":qb_common,"q_bias_centered_common_bias_difference":qb_common-qb,
      "max_abs_centered_score_change_after_common_bias":centered_score_change,
      "max_abs_centered_estimator_change_after_common_bias":common_centered_maxdiff,
      "n_test":len(te),"yearhalf_test_reps":year_n,"yearhalf_ksplit":int(cfg["year_split_diagnostic"]["ksplit"]),
      "yearhalf_valid_splits_total":sum(r["yearhalf_splits_valid"] for r in te),
      "yearhalf_failed_splits_total":sum(r["yearhalf_splits_failed"] for r in te),
      "alpha":alpha,"estimator_only_common_bias":shift,
      "equivariance_max_error_delta":equiv_error,"equivariance_max_width_identity_error":equiv_width,
      "source_sha256":sha(SOURCE_FILE)}
    panelfile=run_root()/"results"/f"fixed_{mode}_panel_{scname}_seed{seed}.csv";write_csv(panelfile,allrows)
    return {"panel":panel,"summaries":summaries,"paired":pairs,"panelfile":str(panelfile)}

def run(mode: str, workers: int) -> None:
    cfg=read_config()
    manifest_path=ROOT/"results"/"fixed_preformal_manifest.json"
    if mode=="formal":
        if cfg["protocol_status"]!="approved":
            raise RuntimeError(f"formal run blocked: protocol_status={cfg['protocol_status']!r}")
        if not manifest_path.exists():
            raise RuntimeError("formal run blocked: missing fixed_preformal_manifest.json; pin approved config/code first")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        expected={"config_sha256":sha(CONFIG),"source_sha256":sha(SOURCE_FILE)}
        mismatches={k:(manifest.get(k),v) for k,v in expected.items() if manifest.get(k)!=v}
        if mismatches:
            raise RuntimeError(f"formal run blocked: preformal hash mismatch {mismatches}")
    else:
        manifest=None
    workers=max(1,min(int(workers),4,int(cfg["execution"]["max_processes"])))
    t0=time.time(); start=datetime.now(timezone.utc).isoformat(); tasks=[(s["name"],int(seed),mode) for s in cfg["design"]["scenarios"] for seed in cfg["design"]["fixed_design_seeds"]]
    if workers==1: results=[run_panel(tk) for tk in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool: results=list(pool.map(run_panel,tasks))
    base=run_root()/"results"; panels=[r["panel"] for r in results];summ=[x for r in results for x in r["summaries"]];pairs=[x for r in results for x in r["paired"]]
    write_csv(base/f"fixed_{mode}_calibration.csv",panels);write_csv(base/f"fixed_{mode}_summary.csv",summ);write_csv(base/f"fixed_{mode}_paired_differences.csv",pairs)
    log={"study_id":cfg["study_id"],"mode":mode,"mode_id":int(cfg["sampling"]["mode_ids"][mode]),
      "protocol_status":cfg["protocol_status"],"started_utc":start,
      "finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.time()-t0,"config_sha256":sha(CONFIG),
      "code_sha256":sha(Path(__file__).resolve()),"source_sha256":sha(SOURCE_FILE),"python":sys.version,
      "numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform(),"workers":workers,
      "blas_threads":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")},
      "preformal_manifest_sha256":sha(manifest_path) if manifest is not None else None,
      "panels":[r["panelfile"] for r in results],"warnings":["Truth-trained bias/scale calibration uses simulation truth and is not deployable without external truth-labeled calibration.",
      "The direct common-factor corrected-intersection comparator shares CR1 multiplier and PSD clipping with the legacy comparator; component-specific two-way CR1 is separately reported.",
      "Corrected CR1 groups region x year intersections; pinned-source covariance is a diagnostic.","Counterfactual is fixed and shared by fit and truth; estimated-Delta uncertainty is outside scope.",
      "Year-half q is a same-test-replicate diagnostic based on full-grid predictions; it is not an independent calibration or conformal guarantee.",
      "Residual permutation bootstrap and state split were not reused."]}
    tmp=base/f"fixed_{mode}_reproducibility.json.tmp";tmp.write_text(json.dumps(log,indent=2,ensure_ascii=False),encoding="utf-8");tmp.replace(base/f"fixed_{mode}_reproducibility.json")
    print(json.dumps({"mode":mode,"elapsed_seconds":log["elapsed_seconds"],"panels":len(results)}),flush=True)

def pin_preformal() -> dict[str,Any]:
    cfg=read_config()
    if cfg["protocol_status"]!="approved":
        raise RuntimeError(f"preformal pin blocked: protocol_status={cfg['protocol_status']!r}; root approval must set it to 'approved'")
    base=run_root()/"results"; out=base/"fixed_preformal_manifest.json"
    if out.exists():
        raise RuntimeError(f"preformal manifest already exists: {out}; inspect it and do not silently replace a pin")
    required={"pilot_summary":base/"fixed_pilot_summary.csv",
      "pilot_calibration":base/"fixed_pilot_calibration.csv",
      "pilot_reproducibility":base/"fixed_pilot_reproducibility.json",
      "preformal_review":ROOT/"docs"/"fixed_design_preformal_review.md"}
    missing=[str(path) for path in required.values() if not path.exists()]
    if missing: raise RuntimeError(f"preformal pin blocked: required pilot/review outputs missing: {missing}")
    payload={"study_id":cfg["study_id"],"protocol_status":cfg["protocol_status"],
      "created_utc":datetime.now(timezone.utc).isoformat(),"config_path":str(CONFIG),
      "code_path":str(Path(__file__).resolve()),"source_path":str(SOURCE_FILE),
      "config_sha256":sha(CONFIG),"code_sha256":sha(Path(__file__).resolve()),
      "source_sha256":sha(SOURCE_FILE),"pilot_artifacts":{k:{"path":str(v),"sha256":sha(v)} for k,v in required.items()},
      "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__}
    tmp=out.with_suffix(".json.tmp");tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8");tmp.replace(out)
    return payload

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--mode",choices=("pilot","formal"),default="pilot")
    ap.add_argument("--workers",type=int,default=4);ap.add_argument("--pin-preformal",action="store_true");a=ap.parse_args()
    try:
        if a.pin_preformal:
            payload=pin_preformal()
            print(json.dumps({k:payload[k] for k in ("config_sha256","code_sha256","source_sha256")},indent=2),flush=True)
        else: run(a.mode,a.workers)
    except Exception as e: print(f"ERROR {type(e).__name__}: {e}\n{traceback.format_exc()}",file=sys.stderr);return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
