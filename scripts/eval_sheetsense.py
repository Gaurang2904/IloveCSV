"""Evaluate IloveCSV on a noisy workbook: suggested rules -> issues -> approve -> export -> diff.

    python scripts/eval_sheetsense.py path/to/noisy.xlsx [out_dir]
Prints every issue, anomaly and suggestion, then approves suggestions with
confidence >= 0.9, exports, and shows exactly which cells changed.
"""

import sys
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

from ilovecsv.pipeline import decide, export, run_checks
from ilovecsv.profiling import profile_workbook
from ilovecsv.reader import read_workbook
from ilovecsv.rules.suggest import suggest_rules

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("reports/eval")
rules = suggest_rules(profile_workbook(read_workbook(src)))
print(rules.split("# Example")[0])
run = run_checks(src, rules)
s = run.summary()
print(f"rows by sheet {s['rows_by_sheet']}  errors {s['errors']}  warnings {s['warnings']}  "
      f"anomalies {s['anomalies']}  score {s['quality_score']}")
print("by rule:", dict(Counter(i.rule_id for i in run.validation.issues)))
for i in run.validation.issues:
    print(f"  {i.severity:7} {i.rule_id:22} {i.cell_ref or i.sheet:16} {i.value!r:32} {i.message}")
for a in run.anomalies:
    print(f"  anomaly {a.detector:22} {a.cell_ref:16} {a.value!r:32} {a.message[:70]}")
print(f"\n{len(run.suggestions)} suggestions:", dict(Counter(x.method for x in run.suggestions)))
decide(run, min_confidence=0.9, approver="eval")
ex = export(run, out)
a = ex.after
print(f"applied {len(ex.apply.applied)}  skipped {len(ex.apply.skipped)}")
for sk in ex.apply.skipped:
    print("  skipped", sk["cell_ref"], sk["method"], sk["reason"])
print(f"after: errors {a['errors']}  warnings {a['warnings']}  score {a['quality_score']}  by rule {a['by_rule']}")

before, after = load_workbook(src), load_workbook(ex.corrected_path)
for ws in before.worksheets:
    w2 = after[ws.title]
    print(f"\n{ws.title}: rows {ws.max_row - 1} -> {w2.max_row - 1}")
    changed = [(c.coordinate, c.value, w2[c.coordinate].value) for row in ws.iter_rows() for c in row
               if c.value != w2[c.coordinate].value]
    print(f"  {len(changed)} cells differ (row deletions shift everything below them)")
