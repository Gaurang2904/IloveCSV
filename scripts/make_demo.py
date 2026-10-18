"""Build the demo datasets and run IloveCSV on them.

    python scripts/make_demo.py
Writes:
  examples/churn_demo.xlsx                  synthetic churn table with planted problems
  reports/demo_churn/                       one-click prepare output (dataset + cleaning report)
  examples/demo_orders.xlsx + rules_orders.yaml, reports/demo/   rule-engine demo (manual review flow)
"""

import shutil
from pathlib import Path

from ilovecsv.pipeline import decide, export, prepare, run_checks
from ilovecsv.synth import RULES_YAML, build_churn_workbook, build_orders_workbook

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    churn = ROOT / "examples" / "churn_demo.xlsx"
    build_churn_workbook(churn)
    out = ROOT / "reports" / "demo_churn"
    shutil.rmtree(out, ignore_errors=True)
    res = prepare(churn, out, target="churn")
    a = res.exported.after
    print(f"churn: {len(res.exported.apply.applied)} safe fixes, unresolved errors {a['errors']}, "
          f"warnings {a['warnings']}; {a['quality_calculation']}")

    xlsx = ROOT / "examples" / "demo_orders.xlsx"
    rules = ROOT / "examples" / "rules_orders.yaml"
    build_orders_workbook(xlsx, n=300, seed=7)
    rules.write_text(RULES_YAML, encoding="utf-8")
    run = run_checks(xlsx, RULES_YAML)
    s = run.summary()
    print(f"orders before: quality {s['quality_score']}%, {s['errors']} errors, {s['warnings']} warnings, "
          f"{s['anomalies']} anomalies, {s['suggestions']} suggestions")
    decide(run, min_confidence=0.85, approver="demo-reviewer", include_row_deletions=True)
    shutil.rmtree(ROOT / "reports" / "demo", ignore_errors=True)
    ex = export(run, ROOT / "reports" / "demo")
    a = ex.after
    print(f"orders after:  quality {a['quality_score']}%, {a['errors']} errors, {a['warnings']} warnings; "
          f"applied {len(ex.apply.applied)}, skipped {len(ex.apply.skipped)}")
