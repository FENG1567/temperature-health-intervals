"""Portable read-only inputs and isolated rerun outputs."""
from pathlib import Path
import os
REPO=Path(os.environ.get('CA_REPO_ROOT',Path(__file__).resolve().parents[1])).resolve()
RUN=Path(os.environ.get('CA_RUN_ROOT',REPO/'outputs/run')).resolve()
_OUTPUTS = REPO/'outputs'
if RUN == REPO or RUN in REPO.parents or (REPO in RUN.parents and RUN != _OUTPUTS and _OUTPUTS not in RUN.parents):
 raise ValueError('CA_RUN_ROOT must be outside the repository (not its ancestor), or inside outputs/')
def source_root():return REPO
def run_root():
 RUN.mkdir(parents=True,exist_ok=True)
 return RUN
def output_path(value):
 p=Path(value).resolve()
 if RUN==p or RUN in p.parents:return p
 try:rel=p.relative_to(REPO)
 except ValueError:return p
 parts=rel.parts
 if parts and (parts[0] in ['results','figures','qa','logs','docs'] or (len(parts)>1 and parts[0] in ['city_case','additional_analyses'] and parts[1] in ['results','figures','qa','logs','docs','tables'])):
  return RUN/rel
 return p
