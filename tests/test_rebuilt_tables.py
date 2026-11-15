from __future__ import annotations
import json, unittest, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"tables"

class RebuiltTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable,str(ROOT/"scripts/build_tables.py"),"--output",str(OUT),"--check"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    def test_all_recomputed_values_match(self):
        report=json.loads((OUT/"rebuild_report.json").read_text())
        self.assertEqual(report["check_status"],"passed")
    def test_year_half_subset(self):
        d=pd.read_csv(OUT/"Table_S1_fixed_method_summaries.csv")
        x=d.loc[d.method=="year_half_split_q_diagnostic","n_planned"]
        self.assertEqual(len(x),18)
        self.assertTrue((x==25).all())
        self.assertFalse(d.duplicated(["scenario","design_seed","method","family","case"]).any())
    def test_report_discloses_copied_tables(self):
        report=json.loads((OUT/"rebuild_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["tables"]["Table_S1_fixed_method_summaries.csv"]["level"],"replication_level_resummary")
        self.assertEqual(report["tables"]["Table_S2_calibration_diagnostics.csv"]["level"],"copied_scientific_metadata_or_deterministic_analysis")
    def test_priority_row_counts(self):
        expected={"Table_S1_fixed_method_summaries.csv":198,"Table_S3_fixed_paired_differences.csv":198,
            "Table_S4a_daily_coverage.csv":24,"Table_S4b_real_paired_differences.csv":18,
            "Table_S4c_quasi_paired_differences.csv":18,"Table_S10_mvn_summary.csv":36,
            "Table_S11_mvn_method_paired.csv":24,"Table_S11b_covariance_paired.csv":18,
            "Table_S11c_stability_summary.csv":16,"Table_S11d_stability_detailed.csv":1440}
        for name,n in expected.items():
            with self.subTest(table=name): self.assertEqual(len(pd.read_csv(OUT/name)),n)
    def test_main_outputs_exist(self):
        for name in ("Table_1_fixed_design.csv","Table_2_daily_coverage.csv","Table_3_mvn_intervals.csv","Table_4_monthly_bound_delta_0.05.csv"):
            with self.subTest(table=name): self.assertTrue((OUT/name).is_file())

if __name__=="__main__": unittest.main()

