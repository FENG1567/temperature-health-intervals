import importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1]
def load(run):
 with patch.dict(os.environ,{'CA_REPO_ROOT':str(R),'CA_RUN_ROOT':str(run)}):
  spec=importlib.util.spec_from_file_location('pathcheck',R/'code/repro_paths.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class PathTests(unittest.TestCase):
 def test_ancestor_rejected(self):
  with self.assertRaises(ValueError):load(R.parent)
 def test_archive_and_code_rejected(self):
  for x in ('results','inputs','code','paper','additional_analyses'):
   with self.subTest(x=x),self.assertRaises(ValueError):load(R/x/'run')
 def test_default_routes_archive(self):
  m=load(R/'outputs/run');self.assertEqual(m.output_path(R/'results/x.csv'),R/'outputs/run/results/x.csv')
 def test_external_routes_archive(self):
  d=R.parent/'external_repro_output';m=load(d);self.assertEqual(m.output_path(R/'city_case/results/x.csv'),d/'city_case/results/x.csv')
