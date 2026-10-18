"""Command-line interface.

    ilovecsv prepare  data.xlsx --target Churn --out prepared/     # one click: clean + report
    ilovecsv prepare  new_batch.csv --config prepared/data_prepare_config.yaml
    ilovecsv profile  data.xlsx
    ilovecsv suggest-rules data.xlsx > rules.yaml
    ilovecsv check    data.xlsx --rules rules.yaml --out reports/ [--approve-min 0.9]
    ilovecsv create-key alice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import decide, export, prepare, run_checks
from .profiling import profile_workbook
from .reader import ReadError, read_workbook
from .report import write_reports
from .rules.schema import RulesError
from .rules.suggest import suggest_rules


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ilovecsv",
                                description="Automated dataset quality and preparation for machine learning")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare", help="one click: validate, apply Safe Auto-Clean, export dataset + report")
    pr.add_argument("file")
    pr.add_argument("--out", default="prepared", help="output directory (default: prepared/)")
    pr.add_argument("--target", default=None, help="label column for supervised learning")
    pr.add_argument("--config", default=None, help="reuse a saved *_prepare_config.yaml")
    pr.add_argument("--remove-exact-duplicates", action="store_true",
                    help="let Safe Auto-Clean delete rows identical in every column (default: review)")
    pr.add_argument("--labels", default=None,
                    help="valid target labels, comma separated (e.g. No,Yes); default: inferred from the data")
    pr.add_argument("--format", action="append", choices=["csv", "parquet"], default=None,
                    help="dataset format; repeat for several (default: csv)")

    pp = sub.add_parser("profile", help="print column profiles as JSON")
    pp.add_argument("file")

    ps = sub.add_parser("suggest-rules", help="print a draft rule file")
    ps.add_argument("file")

    pc = sub.add_parser("check", help="validate, write reports and optionally a corrected file")
    pc.add_argument("file")
    pc.add_argument("--rules", required=True, help="YAML rule file")
    pc.add_argument("--out", default="reports/out", help="output directory")
    pc.add_argument("--approve-min", type=float, default=None,
                    help="approve suggestions with confidence >= this and export a corrected file")
    pc.add_argument("--approve-deletions", action="store_true",
                    help="let --approve-min also remove duplicate rows (off by default)")
    pc.add_argument("--approver", default="cli")
    pc.add_argument("--no-annotate", action="store_true", help="do not highlight/comment changed cells")

    pk = sub.add_parser("create-key", help="create an API user and print the key")
    pk.add_argument("name")

    args = p.parse_args(argv)
    try:
        if args.cmd == "prepare":
            return _prepare(args)
        if args.cmd == "profile":
            profiles = profile_workbook(read_workbook(args.file))
            print(json.dumps([s.to_dict() for s in profiles.values()], indent=2, default=str))
        elif args.cmd == "suggest-rules":
            print(suggest_rules(profile_workbook(read_workbook(args.file))), end="")
        elif args.cmd == "check":
            return _check(args)
        elif args.cmd == "create-key":
            from .api.deps import create_user
            from .api.settings import Settings
            from .db import make_engine, make_sessionmaker

            settings = Settings()
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            with make_sessionmaker(make_engine(settings.db_url))() as db:
                key = create_user(db, args.name)
            print(f"API key for {args.name} (shown once, store it safely):\n{key}")
    except ReadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RulesError as e:
        print(f"error: {e}", file=sys.stderr)
        for err in e.errors:
            where = ".".join(err.get("loc", [])) or (f"line {err.get('line')}" if err.get("line") else "")
            print(f"  - {where}: {err['msg']}", file=sys.stderr)
        return 2
    return 0


def _prepare(args) -> int:
    try:
        res = prepare(args.file, args.out, target=args.target, config=args.config,
                      remove_exact_duplicates=True if args.remove_exact_duplicates else None,
                      target_labels=[x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else None,
                      formats=tuple(args.format or ["csv"]))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run, ex = res.run, res.exported
    a, s = ex.after, run.summary()
    review = sum(1 for x in run.suggestions if x.status == "pending") + sum(1 for x in run.anomalies
                                                                             if x.status == "open")
    rows = ", ".join(f"{k} {len(run.wb.sheets[k].df)} -> {len(ex.after_run.wb.sheets[k].df)}" for k in ex.datasets)
    print(f"{run.wb.filename}: rows {rows}")
    print(f"  Safe Auto-Clean applied {len(ex.apply.applied)} fix(es); {review} item(s) need review")
    print(f"  errors {s['errors']} -> {a['errors']} unresolved, warnings {s['warnings']} -> {a['warnings']} unresolved")
    print(f"  quality: {a['quality_calculation']}")
    for sheet in run.rules.sheets:
        if sheet.target:
            print(f"  target: {sheet.target}")
    report = json.loads(res.report.read_text(encoding="utf-8"))
    for entry in report["ml"]["sheets"]:  # leakage suspects are printed with the needs-review columns below
        if entry.get("missing_target_rows"):
            print(f"  ! {len(entry['missing_target_rows'])} row(s) without a label")
        if entry.get("label_conflicts"):
            print(f"  ! {len(entry['label_conflicts'])} group(s) of identical rows with different labels")
        if entry.get("invalid_label_rows"):
            print(f"  ! {len(entry['invalid_label_rows'])} unrecognised label(s), not guessed "
                  f"(valid: {', '.join((entry.get('target_summary') or {}).get('valid_labels') or [])})")
    for sch in report["columns"]:
        for r in sch.get("needs_review", []):
            print(f"  ! review before using {sch['sheet']}.{r['column']}: {'; '.join(r['reasons'])}")
    print(f"  {'NOT FULLY CLEAN' if a['errors'] or a['warnings'] else 'clean'}; files in {ex.out_dir}:")
    if ex.corrected_path.suffix == ".xlsx":
        print(f"    cleaned workbook (changes highlighted): {ex.corrected_path.name}")
    for files in ex.datasets.values():
        for fmt, name in files.items():
            print(f"    ML-ready dataset ({fmt}): {name}")
    for kind, path in ex.reports.items():
        print(f"    report ({kind}): {path.name}")
    if ex.config_path:
        print(f"    config: {ex.config_path.name}  (reuse with --config on new batches)")
    return 1 if a["errors"] else 0


def _check(args) -> int:
    rules_text = Path(args.rules).read_text(encoding="utf-8")
    run = run_checks(args.file, rules_text)
    s = run.summary()
    out = Path(args.out)
    print(f"{run.wb.filename}: {s['rows']} rows checked on {', '.join(s['sheets_checked'])}")
    print(f"  quality score {s['quality_score']}%  errors {s['errors']}  warnings {s['warnings']}  "
          f"anomalies {s['anomalies']}  suggestions {s['suggestions']}")
    for rule, n in s["by_rule"].items():
        print(f"    {rule:<28} {n}")
    for e in s["rule_errors"]:
        print(f"  ! rule {e['rule_id']} could not run: {e['error']}")
    if args.approve_min is not None:
        n = decide(run, min_confidence=args.approve_min, approver=args.approver,
                   include_row_deletions=args.approve_deletions)
        ex = export(run, out, annotate=not args.no_annotate)
        a = ex.after
        pending_rows = sum(1 for x in run.suggestions if x.action == "delete_row" and x.status == "pending")
        print(f"  approved {n} suggestions; applied {len(ex.apply.applied)}, skipped {len(ex.apply.skipped)}")
        if pending_rows:
            print(f"  {pending_rows} duplicate-row deletion(s) left for review (use --approve-deletions)")
        print(f"  after: quality score {a['quality_score']}%  unresolved errors {a['errors']}  "
              f"unresolved warnings {a['warnings']}")
        print(f"  score: {a['quality_calculation']}")
        print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(a["resolution_counts"].items())))
        print(f"  {'NOT FULLY CLEAN' if a['errors'] or a['warnings'] else 'clean'}: corrected file {ex.corrected_path}")
        paths = ex.reports
    else:
        out.mkdir(parents=True, exist_ok=True)
        paths = write_reports(run, None, out)
    for kind, path in paths.items():
        print(f"  {kind} report: {path}")
    return 1 if s["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
