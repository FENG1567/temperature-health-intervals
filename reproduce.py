"""Portable entry point for evidence checks, paper outputs, and analysis reruns."""
from pathlib import Path
import argparse,os,subprocess,sys,json
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('CA_REPO_ROOT',str(ROOT))
os.environ.setdefault('CA_RUN_ROOT',str(ROOT/'outputs/run'))
os.environ.setdefault('MPLBACKEND','Agg')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
def run(args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','restore-cache','tables','figures','smoke','all','analysis']);p.add_argument('--stage',choices=['fixed','heatday','daily-state','daily-city','quasi-state','quasi-city','mechanism-state','mechanism-city','monthly','sensitivity','mvn']);p.add_argument('--formal',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 if a.command in ['verify','all']:run(['scripts/verify_repository.py'])
 if a.command=='restore-cache':run(['scripts/restore_chunks.py'])
 if a.command in ['tables','all']:run(['scripts/build_tables.py','--output','outputs/tables','--check'])
 if a.command in ['figures','all']:
  run(['code/plot_paper_figures.py','--figure','all','--data-root',ROOT])
 if a.command in ['smoke','all']:run(['scripts/smoke_analysis.py'])
 if a.command=='analysis':
  if not a.stage:p.error('analysis requires --stage')
  stage=a.stage;mode='formal' if a.formal else 'pilot'
  if stage in ['fixed','heatday']:run(['code/fixed_design.py' if stage=='fixed' else 'code/fixed_heatday.py','--mode',mode,'--workers',a.workers])
  elif stage.startswith('daily'):run(['city_case/code/real_panel_audit.py' if stage.endswith('city') else 'code/real_panel_audit.py','--mode',mode,'--workers',a.workers])
  elif stage.startswith('quasi'):
   if not a.formal:p.error('quasi replay uses the formal 1000-per-scenario outcomes; pass --formal')
   run(['code/quasi_baseline.py','--case','city' if stage.endswith('city') else 'state','--workers',a.workers])
  elif stage.startswith('mechanism'):run(['code/mechanism_diagnostic.py','--case-root',ROOT/'city_case' if stage.endswith('city') else ROOT])
  elif stage=='monthly':run(['code/monthly_bound_repair.py'])
  elif stage=='sensitivity':run(['additional_analyses/code/sensitivity_extension.py'])
  elif stage=='mvn':
   cmd=['additional_analyses/code/mvn_interval_replay.py','--mode',mode,'--case','both','--workers',a.workers]
   if a.formal:cmd+=['--formal-approved']
   run(cmd)
 return 0
if __name__=='__main__':raise SystemExit(main())
