"""Refit one formal draw in every fixed-family scenario and both daily cases."""
from pathlib import Path
import csv,importlib.util,json,math,os,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CA_REPO_ROOT',str(ROOT));os.environ.setdefault('CA_RUN_ROOT',str(ROOT/'outputs/smoke'))
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import pandas as pd
import fixed_design as fd
import fixed_heatday as hd
import quasi_baseline as qb
checks=[]
for name,module,prefix in [('temperature',fd,'fixed'),('heatday',hd,'heatday')]:
 cfg=module.read_config()
 for sc in cfg['design']['scenarios']:
  seed=int(cfg['design']['design_seeds'][0]) if 'design_seeds' in cfg['design'] else 11501
  design=module.make_design(cfg,sc,seed)
  engine=module if name=='temperature' else module.engine
  y=engine.draw_y(design,cfg,int(sc['id']),'formal','test',0)
  fit=engine.fit_rep(y,design,cfg)
  old=pd.read_csv(ROOT/f"results/{prefix}_formal_panel_{sc['name']}_seed{seed}.csv")
  row=old.loc[old.stage.eq('test') & old.replicate.eq(0)].iloc[0]
  for key in ['A_hat','A_true','h_corrected']:
   if not np.isclose(float(fit[key]),float(row[key]),rtol=1e-7,atol=1e-5,equal_nan=True):raise AssertionError((name,sc['name'],key,fit[key],row[key]))
  checks.append({'family':name,'scenario':sc['name'],'replicate':0,'passed':True})
for case in ['state','city']:
 state=qb._prepare_case(case);original_rows={(str(x['scenario']),int(x['replicate'])):x for x in state['formal'].to_dict(orient='records')};qb._init_worker({'case_state':state,'original_rows':original_rows})
 scenario=next(iter(state['scenario_means']));fit=qb._one_replay((scenario,0))
 if not fit['converged']:raise AssertionError((case,'fit did not converge'))
 if not fit['match_A_hat_1e_8'] or not fit['match_se_model_1e_8']:raise AssertionError((case,'replayed fit differs'))
 checks.append({'family':'daily','case':case,'scenario':scenario,'replicate':0,'passed':True})
report={'status':'passed','scope':'8 selected formal outcome refits, not a full Monte Carlo rerun','checks':checks}
(ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'outputs/smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
