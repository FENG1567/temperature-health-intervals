import importlib.util,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sensitivity_for_test',R/'additional_analyses/code/sensitivity_extension.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ToleranceTests(unittest.TestCase):
 def check(self,label,expected,actual):
  s={'compared_values':0,'mismatch_count':0,'max_relative_error':0.,'mismatches':[]};m._compare_trees(expected,actual,label,s);return s['mismatch_count']
 def test_nan_scientific_value_fails(self):self.assertEqual(self.check("x.bound",1.,float("nan")),1)
 def test_infinite_scientific_value_fails(self):self.assertEqual(self.check("x.bound",1.,float("inf")),1)
 def test_roundoff_diagnostic(self):self.assertEqual(self.check('x.LP_vertex_objective_max_error',0.,1e-12),0)
 def test_large_diagnostic_error_still_fails(self):self.assertEqual(self.check('x.LP_vertex_objective_max_error',0.,1e-6),1)
 def test_scientific_bound_not_relaxed(self):self.assertEqual(self.check('x.bound',0.,1e-12),1)
if __name__=='__main__':unittest.main()
