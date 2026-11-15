"""Restore optional exact MVN factor caches without Git LFS."""
from pathlib import Path
import argparse,hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def restore(verify_only=False):
 manifest=json.loads((ROOT/'.data_chunks/manifest.json').read_text())
 for entry in manifest['files']:
  target=(ROOT/entry['path']).resolve()
  if ROOT not in target.parents:raise ValueError('Invalid restored path')
  for part in entry['chunks']:
   p=(ROOT/part['path']).resolve()
   if ROOT not in p.parents or digest(p)!=part['sha256']:raise ValueError('Chunk checksum mismatch: '+str(p))
  if verify_only:continue
  if target.exists():
   if digest(target)!=entry['sha256']:raise ValueError('Refusing to overwrite a different cache: '+str(target))
   continue
  target.parent.mkdir(parents=True,exist_ok=True)
  temp=target.with_suffix('.restoring')
  with temp.open('wb') as out:
   for part in entry['chunks']:
    with (ROOT/part['path']).open('rb') as source:
     for block in iter(lambda:source.read(1048576),b''):out.write(block)
  if digest(temp)!=entry['sha256']:raise ValueError('Restored checksum mismatch')
  os.replace(temp,target)
 return {'status':'passed','caches':len(manifest['files']),'mode':'verify' if verify_only else 'restore'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');a=p.parse_args();print(json.dumps(restore(a.verify_only)))
