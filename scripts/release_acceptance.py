"""Run the IloveCSV final-release acceptance pack (T01-T16 + global criteria) and record the results.

    python scripts/release_acceptance.py PATH_TO_EXTRACTED_PACK [--out reports/release_acceptance.md]

Every check is evaluated against the audit reports, manifests and exported files, and the source
files are hashed before and after to prove they are never modified. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_testpack as core  # noqa: E402

from ilovecsv.adapters import inspect_file  # noqa: E402
from ilovecsv.cli import main as cli  # noqa: E402

T_IDS = {"01_noisy_churn.csv": "T01", "02_noisy_churn.xlsx": "T02", "03_noisy_churn.jsonl": "T03",
         "04_conflicting_labels.csv": "T04", "05_sensor_arrays.npz": "T07", "06_synthetic_battery.mat": "T08",
         "07_sensor_data.h5": "T09"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_cli(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli(list(args))
    return code, out.getvalue(), err.getvalue()


def report_of(out: Path) -> dict:
    return json.loads(next(out.glob("*_cleaning_report.json")).read_text(encoding="utf-8"))


def unsupervised(pack: Path, out: Path) -> None:
    check = core.check
    name = "08_unsupervised_customer_segments.csv"
    res, r = core.run(pack / name, out / "08")
    df = pd.read_csv(res.exported.out_dir / next(iter(res.exported.datasets.values()))["csv"], dtype=str,
                     keep_default_na=False)
    check(name, "no target required, no label warnings invented", not r["ml"]["sheets"][0].get("target")
          and not [i for i in r["issues"] if i["rule_id"].startswith("target")])
    check(name, "rows preserved (41 incl. the repeat)", len(df) == 41, f"{len(df)} rows")
    dup = core.issues(r, "row.duplicate")
    check(name, "exact repeated record flagged for review", len(dup) == 1 and dup[0]["resolution"] == "pending_review",
          core.ev(dup[0] if dup else None))
    check(name, "placeholders N/A / unknown -> missing", df.set_index("customer_key").loc["C007", "monthly_spend"] == ""
          and df.iloc[21]["region"] == "", "C007 spend, C022 region")
    for value, col, what in [("twelve hundred", "monthly_spend", "invalid numeric 'twelve hundred'"),
                             ("two", "support_tickets", "invalid numeric 'two'"),
                             ("-250", "monthly_spend", "negative spend"), ("-1", "visits_per_month", "negative visits"),
                             ("99999", "monthly_spend", "extreme spend 99999"), ("99", "support_tickets", "extreme 99")]:
        i = core.find(r, value, col)
        check(name, f"flagged, kept: {what}", i is not None and i["resolution"] == "open", core.ev(i))
    regions = set(df["region"]) - {""}
    check(name, "region casing/whitespace normalised", regions == {"north", "south", "east", "west"}, str(sorted(regions)))
    sch = r["columns"][0]
    check(name, "constant column flagged (kept, not a feature)", sch["excluded"]["constant"] == ["constant_source"]
          and "constant_source" in df.columns, str(sch["excluded"]))
    check(name, "customer_key identified as identifier", sch["excluded"]["identifier"] == ["customer_key"])
    check(name, "feature profiling (distributions, cardinality)", all(c.get("numeric") or c["n_distinct"]
          for p in r["profiles"] for c in p["columns"]), "Profile sheet / profiles[]")

    name = "09_unsupervised_sensor_timeseries.csv"
    res, r = core.run(pack / name, out / "09")
    df = pd.read_csv(res.exported.out_dir / next(iter(res.exported.datasets.values()))["csv"], dtype=str,
                     keep_default_na=False)
    src = pd.read_csv(pack / name, dtype=str, keep_default_na=False)
    check(name, "no labels required", not r["ml"]["sheets"][0].get("target"))
    check(name, "row order and count preserved (no sorting)", list(df["timestamp"]) == list(src["timestamp"]),
          f"{len(src)} -> {len(df)} rows")
    check(name, "missing temperature reported, not interpolated", df.iloc[2]["temperature_c"] == ""
          and any(c["name"] == "temperature_c" and c["missing"] == 1 for c in r["columns"][0]["columns"]))
    for value, what in [("140.0", "temperature 140"), ("15.0", "voltage 15.0"), ("not-a-timestamp", "invalid timestamp")]:
        i = core.find(r, value)
        check(name, f"flagged: {what}", i is not None, core.ev(i))
    gap = core.issues(r, "timeseries.gap")
    check(name, "timestamp gap for S-01 flagged", len(gap) == 1 and "S-01" in gap[0]["message"],
          gap[0]["message"][:90] if gap else "")
    rep = core.issues(r, "timeseries.repeated_reading")
    check(name, "repeated measurement flagged (S-02 10:02 / 10:03)", len(rep) == 1 and "S-02" in rep[0]["message"],
          core.ev(rep[0] if rep else None))
    fw = set(df["firmware"])
    check(name, "firmware casing/whitespace normalised", fw == {"v1.2", "v2.0"}, str(sorted(fw)))


def corrupt(pack: Path, out: Path) -> None:
    check = core.check
    cases = [  # file, expected status without consent, recoverable with --skip-bad-records
        ("10_jsonl_partial_corruption.jsonl", "malformed-records", True),
        ("11_csv_malformed_records.csv", "malformed-records", True),
        ("12_extension_mismatch.csv", "content-mismatch", False),
        ("13_truncated_xlsx.xlsx", "damaged", False),
        ("14_archive_crc_failure.zip", "damaged", True),
        ("15_truncated_npz.npz", "damaged", False),
        ("16_not_a_dataset.bin", "unsupported", False),
    ]
    for name, status, recoverable in cases:
        code, stdout, stderr = run_cli("prepare", str(pack / name), "--out", str(out / f"{name}_strict"))
        check(name, f"refused with status [{status}], nothing exported", code == 2 and f"error [{status}]" in stderr
              and not (out / f"{name}_strict").exists(), stderr.strip().splitlines()[0][:110] if stderr else stdout[:80])
        check(name, "no raw traceback", "Traceback" not in stderr + stdout)
        if recoverable:
            dest = out / f"{name}_recovered"
            code, stdout, _ = run_cli("prepare", str(pack / name), "--skip-bad-records", "--out", str(dest))
            r = report_of(dest)
            excluded = r["source"]["excluded_records"]
            check(name, "consented recovery: output marked PARTIAL, never 'clean'", code == 1 and "PARTIAL IMPORT"
                  in stdout and r["source"]["partial_reason"], r["source"]["partial_reason"][:90])
            check(name, "each excluded record/member listed with its line or member", bool(excluded) and all(
                p.get("line") or p.get("member") for p in excluded),
                "; ".join(f"{p.get('member') or ''} line {p.get('line')}: {p['problem'][:40]}" for p in excluded))
            check(name, "recovered output re-validated", bool(r["verification"]["passed"]))
    insp = inspect_file(pack / "11_csv_malformed_records.csv")
    check("11_csv_malformed_records.csv", "inspection previews the problems with line numbers",
          [p["line"] for p in insp.excluded] == [3, 4], "; ".join(f"line {p['line']}: {p['problem'][:45]}" for p in insp.excluded))
    insp = inspect_file(pack / "14_archive_crc_failure.zip")
    check("14_archive_crc_failure.zip", "verified member listed, damaged member reported",
          [t.name for t in insp.tables] == ["good"] and insp.excluded[0]["member"] == "damaged.csv")


def global_checks(pack: Path, out: Path, before: dict[str, str]) -> None:
    check = core.check
    after = {p.name: sha(p) for p in sorted(pack.iterdir()) if p.is_file()}
    check("(all)", "original uploaded bytes never modified", before == after, f"{len(after)} files re-hashed")
    copies = list(out.rglob("original/*"))
    same = all(sha(c) == before.get(c.name) for c in copies)
    check("(all)", "preserved original copies are byte-identical", bool(copies) and same, f"{len(copies)} copies")
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("*_cleaning_report.json")]
    check("(all)", "every export re-opened and re-validated", all(r["verification"]["passed"] for r in reports),
          f"{len(reports)} reports")
    kinds = {i["resolution"] for r in reports for i in r["issues"]}
    check("(all)", "reports distinguish resolved / open / pending_review", {"resolved", "open", "pending_review"} <= kinds,
          ", ".join(sorted(kinds)))
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("*_manifest.json")]
    not_done = all({"imputation", "scaling", "encoding", "outlier removal", "column dropping"}
                   <= set(m["transformations"]["not_done_by_design"]) for m in manifests)
    rows_ok = all(t["rows_read"] == t["rows_exported"] for m in manifests for t in m["tables"])
    check("(all)", "no imputation / scaling / encoding / outlier or column deletion", not_done)
    check("(all)", "row counts before = after for every table (no silent row loss)", rows_ok,
          f"{sum(len(m['tables']) for m in manifests)} tables")


def markdown(results, pack: Path, seconds: float) -> str:
    failed = [r for r in results if not r[2]]
    lines = ["# IloveCSV release acceptance results", "",
             f"Pack: `{pack.name}` (checksums verified with `SHA256SUMS.txt`). {len(results) - len(failed)} of "
             f"{len(results)} checks passed in {seconds:.0f} s. Generated by `scripts/release_acceptance.py`; every "
             "check reads the audit reports, manifests and exported files, not the console output.", "",
             "| ID | File | Check | Result | Evidence |", "|---|---|---|---|---|"]
    tid = {**T_IDS, "08_unsupervised_customer_segments.csv": "T05", "09_unsupervised_sensor_timeseries.csv": "T06",
           **{f: f"T{int(f[:2])}" for f in ["10_jsonl_partial_corruption.jsonl", "11_csv_malformed_records.csv",
                                             "12_extension_mismatch.csv", "13_truncated_xlsx.xlsx",
                                             "14_archive_crc_failure.zip", "15_truncated_npz.npz", "16_not_a_dataset.bin"]}}
    for file, what, ok, evidence in results:
        ev = str(evidence).replace("|", "\\|")[:160]
        lines.append(f"| {tid.get(file, 'G')} | `{file}` | {what} | {'PASS' if ok else '**FAIL**'} | {ev} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pack", type=Path)
    ap.add_argument("--work", type=Path, default=Path("prepared_release_acceptance"))
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "release_acceptance.md")
    args = ap.parse_args()
    t0 = time.perf_counter()
    before = {p.name: sha(p) for p in sorted(args.pack.iterdir()) if p.is_file()}
    for name in ("01_noisy_churn.csv", "02_noisy_churn.xlsx", "03_noisy_churn.jsonl"):
        core.churn(args.pack, args.work, name)
    core.conflicts(args.pack, args.work)
    unsupervised(args.pack, args.work)
    core.sensors(args.pack, args.work)
    corrupt(args.pack, args.work)
    global_checks(args.pack, args.work, before)
    md = markdown(core.results, args.pack, time.perf_counter() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    failed = [r for r in core.results if not r[2]]
    for file, what, ok, evidence in core.results:
        if not ok:
            print(f"FAIL  {file:38} {what}  [{evidence}]")
    print(f"{len(core.results) - len(failed)}/{len(core.results)} checks passed; results in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
