"""Run IloveCSV on the SheetSense preprocessing test pack and check every expectation in its guide
against the audit reports (not just the exported files).

    python scripts/check_testpack.py PATH_TO_EXTRACTED_TESTPACK [--out prepared_testpack]

Each line is PASS or FAIL with the evidence (cell reference, rule, value) taken from the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ilovecsv.pipeline import prepare  # noqa: E402

results: list[tuple[str, str, bool, str]] = []


def check(file: str, what: str, ok: bool, evidence: str = "") -> None:
    results.append((file, what, bool(ok), evidence))


def issues(report: dict, rule: str | None = None, column: str | None = None, kind: str = "error") -> list[dict]:
    return [i for i in report["issues"] if i["kind"] == kind and (rule is None or i["rule_id"] == rule)
            and (column is None or i.get("column") == column)]


def find(report: dict, value, column: str | None = None, rule: str | None = None) -> dict | None:
    for i in report["issues"]:
        if str(i.get("value")).strip() == str(value) and (column is None or i.get("column") == column) \
                and (rule is None or i["rule_id"] == rule):
            return i
    return None


def ev(i: dict | None) -> str:
    return f"{i['cell_ref']} {i['rule_id']} ({i['resolution']})" if i else "not found"


def run(path: Path, out: Path, **kw):
    res = prepare(path, out, **kw)
    report = json.loads(res.report.read_text(encoding="utf-8"))
    return res, report


def churn(pack: Path, out: Path, name: str) -> None:
    res, r = run(pack / name, out / Path(name).stem, target="churn")
    ex = res.exported
    df = pd.read_csv(ex.out_dir / next(iter(ex.datasets.values()))["csv"], dtype=str, keep_default_na=False)
    src_rows = len(res.run.wb.first_sheet.df)
    check(name, "rows and columns preserved", len(df) == src_rows and {"record_id", "cancellation_date"} <= set(df),
          f"{src_rows} -> {len(df)} rows, {len(df.columns)} columns")
    row = df.set_index("record_id")
    check(name, "trimmed + canonical categories", row.loc["CUST-002", "city"] == "Mumbai"
          and row.loc["CUST-004", "plan"] == "Premium", f"'{row.loc['CUST-002', 'city']}', '{row.loc['CUST-004', 'plan']}'")
    labels = set(df["churn"]) - {""}
    check(name, "target labels normalised", labels == {"No", "Yes"}, f"labels {sorted(labels)}")
    check(name, "numeric strings converted ('1,299', '₹1,299', '999.00')",
          {row.loc["CUST-030", "monthly_spend"], row.loc["CUST-038", "monthly_spend"]} == {"1299"}
          and row.loc["CUST-045", "monthly_spend"] == "999",
          f"{row.loc['CUST-030', 'monthly_spend']}, {row.loc['CUST-038', 'monthly_spend']}, {row.loc['CUST-045', 'monthly_spend']}")
    check(name, "dates converted ('18 Sep 2025', '2025/08/17')",
          row.loc["CUST-020", "last_login_date"] == "2025-09-18" and row.loc["CUST-022", "last_login_date"] == "2025-08-17")
    blanks = {k: row.loc[k, c] for k, c in [("CUST-009", "age"), ("CUST-010", "age"), ("CUST-014", "city"),
                                            ("CUST-015", "plan"), ("CUST-017", "monthly_spend"),
                                            ("CUST-019", "support_tickets_90d"), ("CUST-041", "city")]}
    check(name, "placeholders (N/A, unknown, -) -> missing, nothing imputed", all(v == "" for v in blanks.values()),
          str(blanks))
    for value, col, rule, what in [("not-an-email", "email", "format.email", "malformed email"),
                                   ("2025-02-30", "last_login_date", "type", "impossible date"),
                                   ("07/08/2025", "last_login_date", "type.date_as_text", "ambiguous date"),
                                   ("-4", "age", "range", "age -4"), ("151", "age", "range", "age 151"),
                                   ("Hydrabad", "city", "allowed", "unknown alias 'Hydrabad'"),
                                   ("bangalore", "city", "allowed", "alias 'bangalore'"),
                                   ("two", "support_tickets_90d", "type", "non-numeric 'two'"),
                                   ("12 months", "tenure_months", "type", "non-numeric '12 months'")]:
        i = find(r, value, col, rule)
        unresolved_ok = i is not None and i["resolution"] in ("open", "pending_review")
        check(name, f"flagged: {what}", unresolved_ok, ev(i))
    sug = next((s for s in r["review_required"] if s.get("old") == "Hydrabad"), None)
    check(name, "'Hydrabad' -> 'Hyderabad' suggested for review, not applied", sug is not None and sug["new"] == "Hyderabad"
          and row.loc["CUST-026", "city"] == "Hydrabad", f"{sug and sug['method']}")
    amb = next((s for s in r["review_required"] if s.get("old") == "07/08/2025"), None)
    check(name, "ambiguous date left for review", amb is not None and row.loc["CUST-031", "last_login_date"] == "07/08/2025")
    ml = r["ml"]["sheets"][0]
    ts = ml["target_summary"]
    check(name, "missing targets detected + class distribution", ts["missing"] == 6 and ts["classes"],
          f"missing {ts['missing']}, classes {[(c['label'], c['count']) for c in ts['classes']]}")
    sch = r["columns"][0]
    check(name, "record_id = identifier, email = PII", sch["excluded"]["identifier"] == ["record_id"]
          and sch["excluded"]["pii"] == ["email"], f"{sch['excluded']}")
    leak = [x for x in sch["needs_review"] if x["column"] == "cancellation_date"]
    check(name, "cancellation_date flagged as leakage, column kept", bool(leak) and "cancellation_date" in df.columns,
          leak[0]["reasons"][0][:90] if leak else "")
    dup = issues(r, "row.duplicate")
    check(name, "duplicate record reported for review, not removed", len(dup) == 1 and dup[0]["resolution"] ==
          "pending_review" and (df["record_id"] == "CUST-001").sum() == 2, ev(dup[0] if dup else None))
    not_done = json.loads(ex.manifest_path.read_text())["transformations"]["not_done_by_design"]
    check(name, "no imputation / scaling / encoding / column or outlier deletion", {"imputation", "scaling", "encoding",
          "outlier removal", "column dropping"} <= set(not_done) and len(df.columns) >= 11)
    check(name, "exported file re-validated", bool(r["verification"]["passed"]),
          f"{len(r['verification']['checks'])} checks; unresolved errors {r['after']['errors']}")


def conflicts(pack: Path, out: Path) -> None:
    name = "04_conflicting_labels.csv"
    res, r = run(pack / name, out / "04", target="churn")
    conf = issues(r, "target.conflict", kind="error")
    rows = sorted({i["cell_ref"] for i in conf})
    check(name, "explicit warning for EDGE-001 / EDGE-002 conflict", len(conf) == 2, f"{rows}: {conf[0]['message'] if conf else ''}")
    miss = issues(r, "target.missing")
    check(name, "EDGE-007 missing target flagged, row kept", len(miss) == 1 and miss[0]["cell_ref"].endswith("H8"),
          ev(miss[0] if miss else None))
    ts = r["ml"]["sheets"][0]["target_summary"]
    check(name, "class counts + missing labels reported", ts["missing"] == 1,
          f"{[(c['label'], c['count']) for c in ts['classes']]}, missing {ts['missing']}")


def sensors(pack: Path, out: Path) -> None:
    name = "05_sensor_arrays.npz"
    res, r = run(pack / name, out / "05", source={"align_arrays": True})
    tables = r["source"]["tables"]
    check(name, "arrays inspected; names array kept as metadata", "sensor_names" not in tables and any(
        u["path"] == "sensor_names" for u in r["source"]["unsupported"]), f"tables {list(tables)}")
    miss = {c["name"]: c["missing"] for s in r["columns"] for c in s["columns"] if c["missing"]}
    check(name, "NaN reported (voltage, current)", miss.get("voltage") == 1 and miss.get("current") == 1, str(miss))
    check(name, "implausible voltage 12.5 flagged", find(r, 12.5, "voltage") is not None, ev(find(r, 12.5, "voltage")))
    check(name, "implausible temperature 130 flagged", find(r, 130.0, "temperature") is not None,
          ev(find(r, 130.0, "temperature")))
    check(name, "duplicated cycle index flagged", find(r, 6, "cycle_index", "unique") is not None,
          ev(find(r, 6, "cycle_index", "unique")))
    _, r2 = run(pack / name, out / "05_default")
    check(name, "alignment not assumed by default (hint instead)", "05_sensor_arrays" not in r2["source"]["tables"]
          and any("align_arrays" in w for w in r2["source"]["warnings"]), f"tables {list(r2['source']['tables'])}")

    name = "06_synthetic_battery.mat"
    res, r = run(pack / name, out / "06")
    t = r["source"]["tables"]
    check(name, "nested struct inspected; cycle identity kept", {"BatteryDemo.cycle", "BatteryDemo.cycle.series"} <= set(t),
          f"tables {list(t)}")
    w = [x for x in r["source"]["warnings"] if "lengths differ" in x]
    check(name, "cycle 2 length mismatch flagged", bool(w), w[0][:110] if w else "")
    check(name, "temperature 130 flagged for review, not deleted",
          find(r, 130.0, "Temperature_measured") is not None and len(res.run.wb.sheets["BatteryDemo.cycle.series"].df) == 9,
          ev(find(r, 130.0, "Temperature_measured")))

    name = "07_sensor_data.h5"
    res, r = run(pack / name, out / "07", source={"align_arrays": True})
    struct = r["source"]["structure"]
    check(name, "datasets, shapes, dtypes and metadata listed", any(n.get("attrs") for n in struct) and len(struct) >= 5,
          f"{len(struct)} nodes; root attrs {struct[0].get('attrs')}")
    miss = {c["name"]: c["missing"] for s in r["columns"] for c in s["columns"] if c["missing"]}
    check(name, "NaN reported", miss.get("temperature_c") == 1, str(miss))
    check(name, "extreme temperature 140 / voltage 15 flagged", find(r, 140.0) is not None and find(r, 15.0) is not None,
          f"{ev(find(r, 140.0))}; {ev(find(r, 15.0))}")


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print '₹'
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pack", type=Path)
    ap.add_argument("--out", type=Path, default=Path("prepared_testpack"))
    args = ap.parse_args()
    for name in ("01_noisy_churn.csv", "02_noisy_churn.xlsx", "03_noisy_churn.jsonl"):
        churn(args.pack, args.out, name)
    conflicts(args.pack, args.out)
    sensors(args.pack, args.out)
    for file, what, ok, evidence in results:
        print(f"{'PASS' if ok else 'FAIL'}  {file:28} {what}" + (f"  [{evidence}]" if evidence else ""))
    failed = sum(not ok for *_, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed; reports in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
