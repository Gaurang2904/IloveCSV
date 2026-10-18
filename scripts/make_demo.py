"""Build the demo workbook + rule file, then run the full check and export.

    python scripts/make_demo.py
Writes examples/demo_orders.xlsx, examples/rules_orders.yaml and reports/demo/*.
"""

from pathlib import Path

from ilovecsv.pipeline import decide, export, run_checks
from ilovecsv.synth import RULES_YAML, build_orders_workbook

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    xlsx = ROOT / "examples" / "demo_orders.xlsx"
    rules = ROOT / "examples" / "rules_orders.yaml"
    build_orders_workbook(xlsx, n=300, seed=7)
    rules.write_text(RULES_YAML, encoding="utf-8")
    run = run_checks(xlsx, RULES_YAML)
    s = run.summary()
    print(f"before: quality {s['quality_score']}%, {s['errors']} errors, {s['warnings']} warnings, "
          f"{s['anomalies']} anomalies, {s['suggestions']} suggestions")
    decide(run, min_confidence=0.85, approver="demo-reviewer", include_row_deletions=True)
    ex = export(run, ROOT / "reports" / "demo")
    a = ex.after
    print(f"after:  quality {a['quality_score']}%, {a['errors']} errors, {a['warnings']} warnings; "
          f"applied {len(ex.apply.applied)}, skipped {len(ex.apply.skipped)}")
    for k, p in ex.reports.items():
        print(f"{k}: {p.relative_to(ROOT)}")
