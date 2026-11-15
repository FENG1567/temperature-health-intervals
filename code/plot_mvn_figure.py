"""Render the predeclared daily MVN coverage comparison as Figure 5."""
from pathlib import Path
import csv, hashlib, json, os, sys
from datetime import datetime, timezone
_REPRO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "reproduce.py").is_file())
sys.path.insert(0, str(_REPRO_ROOT / "code"))
from repro_paths import source_root, run_root
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import pandas as pd
from PIL import Image

REPO = source_root()
ROOT = REPO / "additional_analyses"
FIG = run_root() / "figures"
QA = run_root() / "qa"
FIG.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
rows=[]; source_paths=[]
for case in ["state", "city"]:
    p=ROOT/f"tables/mvn_formal_{case}_summary.csv"
    rows += list(csv.DictReader(p.open(encoding="utf-8-sig")))
    source_paths.append(p)
PRIVATE_FONTS = Path(
    os.environ.get("CA_FIGURE_FONT_DIR", Path(__file__).resolve().parents[1] / "fonts")
).expanduser()
for font_name in ("arial.ttf", "arialbd.ttf"):
    font_path = PRIVATE_FONTS / font_name
    if font_path.is_file():
        try:
            font_manager.fontManager.addfont(str(font_path))
        except (OSError, RuntimeError):
            pass
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans","sans-serif"],"font.size":8.2,"font.weight":"bold","axes.labelsize":8.2,"axes.labelweight":"bold","axes.titlesize":10,"axes.titleweight":"bold","figure.titleweight":"bold","xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"mathtext.default":"bf","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white","savefig.bbox":None})
fig,axs=plt.subplots(2,3,figsize=(7.20,5.0),sharey=True)
methods=["normal_wald","linear_gaussian_mc","exact_nonlinear_mc"]
colors=["#747D82","#0072B2","#D88300"]
markers=["o","s","^"]
scenarios=["model_compatible","smooth_omitted_interaction","correlated_overdispersed_ar1"]
names=["Compatible","Omitted interaction","Correlated AR(1)"]
case_names={"state":"State","city":"Municipality"}
method_names=["Normal Wald","Linear Gaussian","Exact nonlinear"]
for i,case in enumerate(["state","city"]):
    for j,sc in enumerate(scenarios):
        ax=axs[i,j]
        for k,m in enumerate(methods):
            group=[next(x for x in rows if x["case"]==case and x["scenario"]==sc and x["covariance_method"]==v and x["interval_method"]==m) for v in ["quasi_pearson","hac28"]]
            ax.errorbar([0+(k-1)*.14,1+(k-1)*.14],[100*float(x["coverage_planned"]) for x in group],yerr=[100*float(x["coverage_mcse_planned"]) for x in group],linestyle="none",marker=markers[k],color=colors[k],capsize=2.5,markersize=4,label=method_names[k])
        ax.axhline(90,color="#4A5358",ls="--",lw=.8)
        ax.set_ylim(60,100); ax.set_xlim(-.4,1.4)
        ax.set_xticks([0,1],["Quasi-Poisson","HAC28"])
        ax.grid(axis="y",alpha=.18)
        panel_title = f"{case_names[case]} | {names[j]}"
        if case == "city":
            panel_title = f"Municipality\n{names[j]}"
        ax.set_title(panel_title,fontsize=10,pad=3,fontweight="bold")
        ax.text(-.13,1.04,"ABCDEF"[i*3+j],transform=ax.transAxes,fontweight="bold",fontsize=18)
        if j==0: ax.set_ylabel("Coverage (%)")
        if i==0: ax.tick_params(axis="x",labelsize=8)
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.92),ncol=3,frameon=False,columnspacing=1.1,handletextpad=.45)
fig.suptitle("Matched covariance coverage",y=.99,fontsize=10,fontweight="bold")
fig.subplots_adjust(left=.10,right=.99,bottom=.12,top=.78,hspace=.52,wspace=.22)
for artist in fig.findobj(match=Text):
    if artist.get_text().strip():
        artist.set_fontweight("bold")
paths=[FIG/f"Figure_5.{ext}" for ext in ["pdf","png","svg","tiff"]]
fig.savefig(FIG / "Figure_5.pdf",bbox_inches=None)
fig.savefig(FIG / "Figure_5.png",dpi=300,bbox_inches=None)
fig.savefig(FIG / "Figure_5.svg",bbox_inches=None)
fig.savefig(FIG / "Figure_5.tiff",dpi=600,bbox_inches=None,pil_kwargs={"compression":"tiff_lzw"})
with Image.open(paths[3]) as image: image.convert("RGB").save(paths[3],compression="tiff_lzw",dpi=(600,600))
plt.close(fig)
source_csv=QA/"figure_sources_mvn.csv"
with source_csv.open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
source_json=QA/"figure_sources_mvn.json"
source_json.write_text(json.dumps({"sources":[{"file":str(p.relative_to(REPO)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],"rows":len(rows),"n_per_point":1000,"target":"same fixed-path expected-count contrast","validity":"all reported method intervals valid","error_bars":"one binomial MCSE"},indent=2),encoding="utf-8")
def metadata(path): return {"file":str(path.relative_to(run_root())),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
manifest_path=QA/"figure_manifest.json"
manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"schema_version":1,"figures":{}}
manifest["generated_at_utc"]=datetime.now(timezone.utc).isoformat()
manifest["runtime"]={"python":sys.version.split()[0],"matplotlib":matplotlib.__version__,"pandas":pd.__version__,"backend":matplotlib.get_backend()}
manifest.setdefault("figures",{})["figure_5_daily_mvn"]={"outputs":[metadata(p) for p in paths],"sources":[{"file":str(p.relative_to(REPO) if p.is_relative_to(REPO) else p.relative_to(run_root())),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in [*source_paths,source_csv,source_json]],"png_dpi":300,"tiff_dpi":600,"tiff_compression":"LZW","tiff_color":"RGB","vector_text":"SVG text remains text; PDF uses TrueType font embedding (fonttype 42)."}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
contract_path=run_root()/"docs/figure_contracts.json"
contract=json.loads(contract_path.read_text(encoding="utf-8"))
contract["figures"]["figure_5_daily_mvn"]["status"]="rendered; visual QA pending"
contract_path.write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Figure 5 exported to PDF/PNG/SVG/TIFF; {len(rows)} formal rows retained.")
