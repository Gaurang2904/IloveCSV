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
    for stream in (sys.stdout, sys.stderr):  # a Windows console (cp1252) cannot print '→' or 'é'
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    p = argparse.ArgumentParser(prog="ilovecsv",
                                description="Universal dataset inspection and safe preparation for machine learning")
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
    _source_args(pr)

    pi = sub.add_parser("inspect", help="structure first: tables, rows, columns, samples, documentation")
    pi.add_argument("file")
    pi.add_argument("--json", action="store_true", help="print the inspection as JSON")
    _source_args(pi)

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
        if args.cmd == "inspect":
            return _inspect(args)
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


def _source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("how the file becomes tables (saved in the recipe)")
    g.add_argument("--sheet", action="append", default=None, help="Excel sheet to process; repeat for several")
    g.add_argument("--table", action="append", default=None,
                   help="table to process from an archive / MATLAB / HDF5 file; repeat for several")
    g.add_argument("--json-records", default=None, help="dot path to the records array in a JSON file")
    g.add_argument("--max-rows", type=int, default=None, help="process only the first N rows (a sample)")
    g.add_argument("--delimiter", default=None, help="CSV delimiter (default: detected)")
    g.add_argument("--encoding", default=None, help="text encoding (default: utf-8, then cp1252)")
    g.add_argument("--mat-layout", choices=["auto", "nasa_battery", "generic"], default=None)
    g.add_argument("--align-arrays", action="store_true", default=None,
                   help="combine same-length 1-D arrays into one table (only if you know they are aligned)")


def _source_from(args) -> dict | None:
    opts = {"sheets": args.sheet, "tables": args.table, "json_records": args.json_records,
            "max_rows": args.max_rows, "delimiter": args.delimiter, "encoding": args.encoding,
            "mat_layout": args.mat_layout, "align_arrays": args.align_arrays}
    opts = {k: v for k, v in opts.items() if v is not None}
    return opts or None


def _inspect(args) -> int:
    from .adapters import inspect_file
    from .sources import SourceOptions

    insp = inspect_file(args.file, options=SourceOptions(**(_source_from(args) or {})), sibling_docs=True)
    if args.json:
        print(json.dumps(insp.to_dict(), indent=2, default=str))
        return 0
    print(f"{insp.filename}: {insp.format}, {insp.size_bytes / 1e6:.1f} MB; "
          f"estimated memory to process: {insp.memory_estimate_mb:,.0f} MB")
    for t in insp.tables:
        rows = f"{t.rows:,}" if t.rows is not None else "?"
        print(f"  table {t.name}: {rows} rows{'' if t.rows_exact else ' (estimated)'} x {len(t.columns)} columns"
              + (f"  [{t.source}]" if t.source else ""))
        print("    columns: " + ", ".join(f"{c['name']} ({c['kind']})" for c in t.columns[:12])
              + (" ..." if len(t.columns) > 12 else ""))
        for n in t.notes[:3]:
            print(f"    note: {n}")
    if not insp.full_processing_ok:
        print(f"  ! some tables exceed the {insp.row_limit:,}-row processing limit: use --max-rows N for a sample")
    for u in insp.unsupported:
        print(f"  ! not converted: {u['path']}: {u['reason']}")
    for d in insp.docs:
        print(f"  doc: {d['path']} ({d['kind']})")
        for h in d.get("missing_value_hints", []):
            print(f"    possible missing-value code {h['token']!r}: \"{h['context'][:90]}\"")
    for w in insp.warnings:
        print(f"  ! {w}")
    if insp.tree and (not insp.tables or insp.format not in ("csv", "xlsx", "json", "jsonl", "parquet")):
        print("  structure:")
        for node in insp.tree[:40]:
            attrs = node.get("attrs")
            print(f"    {'  ' * node.get('depth', 0)}{node['path']}: {node['kind']} {node['shape']}"
                  + (f"  {attrs}" if attrs else ""))
        if len(insp.tree) > 40:
            print(f"    ... {len(insp.tree) - 40} more")
    return 0


def _prepare(args) -> int:
    try:
        res = prepare(args.file, args.out, target=args.target, config=args.config,
                      remove_exact_duplicates=True if args.remove_exact_duplicates else None,
                      target_labels=[x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else None,
                      formats=tuple(args.format or ["csv"]), source=_source_from(args))
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
    for w in run.wb.warnings[:6]:
        print(f"  ! source: {w}")
    for u in run.wb.unsupported[:6]:
        print(f"  ! not converted: {u['path']}: {u['reason']}")
    for sch in report["columns"]:
        missing = [f"{c['name']} {c['missing']}" for c in sch["columns"] if c.get("missing")]
        if missing:
            print(f"  missing values in {sch['sheet']} (kept, never imputed): {', '.join(missing[:12])}")
    print(f"  {'NOT FULLY CLEAN' if a['errors'] or a['warnings'] else 'clean'}; files in {ex.out_dir}:")
    if ex.corrected_path.suffix == ".xlsx":
        print(f"    cleaned workbook (changes highlighted): {ex.corrected_path.name}")
    elif ex.corrected_path.suffix == ".zip":
        print(f"    all cleaned tables (bundle): {ex.corrected_path.name}")
    for files in ex.datasets.values():
        for fmt, name in files.items():
            print(f"    ML-ready dataset ({fmt}): {name}")
    for kind, path in ex.reports.items():
        print(f"    report ({kind}): {path.name}")
    for label, path in (("manifest", ex.manifest_path), ("feature schema", ex.schema_path),
                        ("recipe", ex.config_path), ("reproduce", ex.script_path)):
        if path:
            print(f"    {label}: {path.name}")
    v = ex.verification or {}
    failed = [c for c in v.get("checks", []) if not c["passed"]]
    print(f"  verification of the exported files: {'passed' if v.get('passed') else 'FAILED'} "
          f"({len(v.get('checks', [])) - len(failed)}/{len(v.get('checks', []))} checks)")
    for c in failed:
        print(f"    ! {c['check']}: {c['detail']}")
    if run.recipe_decisions:
        d = run.recipe_decisions
        print(f"  recipe decisions: {d['applied']} of {d['recorded']} applied ({d['not_applicable']} did not match "
              "this data)")
    if run.wb.partial:
        print(f"  ! PARTIAL: only the first {run.wb.partial.get('rows')} rows of each table were processed")
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
