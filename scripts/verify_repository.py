"""Verify archived evidence and GitHub file-size compatibility."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text())
 for item in manifest['files']:
  p=ROOT/item['file']
  if not p.is_file() or digest(p)!=item['sha256']:raise AssertionError('File mismatch: '+item['file'])
 oversized=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and not any(t in p.relative_to(ROOT).parts for t in ['.git','outputs','.venv','__pycache__']) and p.stat().st_size>25*1024*1024 and not p.name.endswith('_factors.npz')]
 if oversized:raise AssertionError('Files exceed browser upload size: '+str(oversized))
 required=['inputs/code_snapshot/core_sim.py','inputs/heatday_snapshot/core_hd.py','inputs/public/bra_state_day.parquet','city_case/inputs/public/sao_paulo_city_day.parquet','inputs/monthly/BRA.csv','inputs/monthly/MEX.csv','inputs/monthly/SIM.csv','inputs/monthly/MEXD.csv']
 for f in required:
  if not (ROOT/f).is_file():raise AssertionError('Missing scientific input: '+f)
 out={'status':'passed','checked_files':len(manifest['files']),'missing_inputs':0,'oversized_upload_files':oversized}
 (ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'outputs/integrity.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':print(json.dumps(verify()))
