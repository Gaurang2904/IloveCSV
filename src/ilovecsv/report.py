"""Cleaning report: one data structure, three renderings (JSON, Excel, HTML).

The JSON form is the machine-readable contract (``schema_version``) for downstream code: what
was changed or removed and why, what still needs review, what is unresolved, and the feature
schema of the cleaned dataset. Every report carries provenance: SHA-256 of the input, the rules
and the outputs, the tool version, timestamps, and who approved each change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .explain import explain_issue
from .ml import feature_schema, mark_needs_review, ml_section
from .models import ANOMALY, ERROR, RESOLUTIONS, jsonable
from .pipeline import now_iso, tool_info
from .reader import partial_text, sha256_file
from .reader import where as record_where

if TYPE_CHECKING:
    from .pipeline import ExportResult, RunResult


SCHEMA_VERSION = "1.0"


def build_audit(run: RunResult, exported: ExportResult | None = None) -> dict[str, Any]:
    errors = [i for i in run.validation.issues if i.kind == ERROR]
    anomalies = [i for i in run.anomalies if i.kind == ANOMALY]
    schema_run = exported.after_run if exported and exported.after_run else run
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_info(),
        "run": {"id": run.run_id, "created_at": run.created_at, "exported_at": now_iso() if exported else None},
        "input": {
            "filename": run.wb.filename,
            "file_type": run.wb.file_type,
            "sha256": run.wb.sha256,
            "sheets": [
                {"name": s.name, "rows": len(s.df), "columns": len(s.df.columns), "header_row": s.header_row,
                 "dropped_blank_rows": s.dropped_blank_rows, "formula_cells": len(s.formula_cells)}
                for s in run.wb.sheets.values()
            ],
            "warnings": run.wb.warnings,
        },
        "rules": {"sha256": run.rules_sha256, "text": run.rules_text},
        "summary": run.summary(),
        "profiles": [p.to_dict() for p in run.profiles.values()],
        "issues": [{**i.to_dict(), "why": explain_issue(i)} for i in errors],
        "anomalies": [{**i.to_dict(), "why": explain_issue(i)} for i in anomalies],
        "suggestions": [s.to_dict() for s in run.suggestions],
        "changes": [],
        "skipped_changes": [],
        "after": None,
        "unresolved": [],
        "output": None,
        "not_verified": not_verified_columns(run),
        "config": {
            "mode": "safe_auto_clean" if run.auto_cleaned else "manual_review",
            "targets": {(s.sheet or run.wb.first_sheet.name): s.target for s in run.rules.sheets if s.target},
            "exact_duplicates": {(s.sheet or run.wb.first_sheet.name): s.duplicates.exact for s in run.rules.sheets},
            "rules_sha256": run.rules_sha256,
        },
        "dataset": None,
        "columns": feature_schema(schema_run),
        "ml": ml_section(run, schema_run),
        "removed_rows": [],
        "review_required": review_queue(run),
        "source": {"format": run.wb.source_format or run.wb.file_type,
                   "options": run.rules.source.model_dump(exclude_defaults=True),
                   "tables": {n: m for n, m in run.wb.tables_meta.items()},
                   "documentation": run.wb.docs, "unsupported": run.wb.unsupported, "partial": run.wb.partial,
                   "warnings": run.wb.warnings, "structure": run.wb.tree[:200],
                   "excluded_records": run.wb.excluded_records,
                   "partial_reason": partial_text(run.wb.partial)},
        "recipe_decisions": run.recipe_decisions,
        "verification": exported.verification if exported else None,
    }
    if exported:
        to_cleaned = exported.to_cleaned_row
        audit["changes"] = [{**c, "row_cleaned": to_cleaned.get(c.get("sheet"), {}).get(c.get("row"))}
                            for c in exported.apply.applied if c["action"] == "set"]
        audit["removed_rows"] = removed_rows(run, exported)
        audit["skipped_changes"] = exported.apply.skipped
        audit["after"] = {k: v for k, v in exported.after.items() if k != "unresolved"}
        audit["unresolved"] = [{**u, "row_original": _row_of(u.get("original_cell_ref")),
                                "row_cleaned": _row_of(u.get("corrected_cell_ref"))}
                               for u in exported.after.get("unresolved", [])]
        audit["output"] = {"filename": exported.corrected_path.name, "sha256": exported.apply.sha256}
        out_dir = exported.out_dir or exported.corrected_path.parent
        audit["input"]["preserved_copy"] = (str(exported.original_copy.relative_to(out_dir))
                                            if exported.original_copy else None)
        audit["dataset"] = {
            "rows_before": {s: len(run.wb.sheets[s].df) for s in exported.datasets},
            "rows_after": {s: len(exported.after_run.wb.sheets[s].df) for s in exported.datasets}
            if exported.after_run else {},
            "files": exported.datasets,
            "sha256": {s: {fmt: sha256_file(out_dir / name) for fmt, name in files.items()}
                       for s, files in exported.datasets.items()},
            "kept_as_text": exported.kept_as_text,
            "config_file": exported.config_path.name if exported.config_path else None,
        }
    mark_needs_review(audit["columns"], audit["ml"], (audit["dataset"] or {}).get("kept_as_text", []))
    audit["findings"] = findings_overview(audit)
    return audit


_FINDING_GROUPS = [  # (category, rule ids or prefixes), first match wins
    ("labels", ("target.",)),
    ("time series", ("timeseries.",)),
    ("duplicates", ("row.duplicate", "unique", "key.duplicate")),
    ("suspicious readings", ("range", "not_future")),
    ("representation (safe conversions)", ("type.stored_as_text", "type.date_as_text", "type.date_serial",
                                           "format.whitespace", "format.null_token", "allowed.noncanonical",
                                           "allowed.alias", "format.email_case", "format.phone_style",
                                           "format.phone_numeric", "format.case_variant")),
]  # everything else (type, allowed, format.email, format.phone, required, regex, length, ...) is "invalid values"


def _category(rule_id: str) -> str:
    for category, ids in _FINDING_GROUPS:
        if any(rule_id.startswith(x) if x.endswith(".") else rule_id == x for x in ids):
            return category
    return "invalid values"


def findings_overview(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per kind of finding, grouped the way a reviewer checks them: structure, missing values,
    invalid values, suspicious readings, duplicates, labels, conversions applied, review queue.
    An empty category means nothing of that kind was found, not that it was not checked."""
    rows: list[dict[str, Any]] = []

    def add(category: str, finding: str, count: int, where: str = "", detail: str = "") -> None:
        rows.append({"category": category, "finding": finding, "count": count, "where": where[:500],
                     "detail": detail[:500]})

    src = audit.get("source") or {}
    for p in src.get("excluded_records", []):
        add("structure", "excluded on request, not imported", 1, record_where(p),
            p["problem"] + (f"  Raw: {p['raw']}" if p.get("raw") else ""))
    for w in src.get("warnings", []):
        add("structure", "source warning", 1, "", w)
    for u in src.get("unsupported", []):
        meta = str(u.get("reason", "")).startswith(("metadata", "single value"))
        add("structure", "metadata (kept in the manifest)" if meta else "not converted to a table", 1,
            u.get("path", ""), u.get("reason", ""))
    for d in src.get("documentation", []):
        hints = ", ".join(str(h.get("token")) for h in d.get("missing_value_hints", []))
        if hints:
            add("structure", "documentation mentions missing-value codes", 1, d.get("path", ""),
                f"{hints} (not applied automatically: add them to null_values if they apply)")
    for sch in audit.get("columns", []):
        for c in sch.get("columns", []):
            if c.get("missing"):
                add("missing values", "missing values (kept, never imputed)", c["missing"],
                    f"{sch['sheet']}.{c['name']}", f"{c.get('missing_rate', 0):.1%} of rows")
    groups: dict[tuple[str, str], list[dict]] = {}
    for i in audit.get("issues", []):
        cat = "suspicious readings" if i.get("kind") == "anomaly" else _category(i.get("rule_id", ""))
        groups.setdefault((cat, i.get("rule_id", "")), []).append(i)
    for (cat, rid), items in sorted(groups.items()):
        where = ", ".join(str(i.get("cell_ref") or "") for i in items[:12]) + (" ..." if len(items) > 12 else "")
        values = ", ".join(repr(i.get("value")) for i in items[:8] if i.get("value") is not None)
        add(cat, rid, len(items), where, (items[0].get("message") or "") + (f" Values: {values}" if values else ""))
    for sch in audit.get("columns", []):
        for col in sch.get("excluded", {}).get("constant", []):
            add("profile", "constant column (one value; kept, not a feature)", 1, f"{sch['sheet']}.{col}",
                "every row holds the same value, so it carries no information for a model")
        for r in sch.get("needs_review", []):
            add("labels" if any("leakage" in x for x in r["reasons"]) else "invalid values",
                "review before using as a feature", 1, f"{sch['sheet']}.{r['column']}", "; ".join(r["reasons"]))
    methods: dict[str, int] = {}
    for c in audit.get("changes", []):
        methods[c["method"]] = methods.get(c["method"], 0) + 1
    for m, n in sorted(methods.items()):
        add("conversions applied", m, n)
    if audit.get("removed_rows"):
        add("conversions applied", "rows removed", len(audit["removed_rows"]))
    if audit.get("review_required"):
        add("review queue", "items waiting for a person", len(audit["review_required"]))
    return rows


def _row_of(ref: str | None) -> int | None:
    if not ref or "!" not in ref:
        return None
    digits = "".join(ch for ch in ref.split("!", 1)[1] if ch.isdigit())
    return int(digits) if digits else None


def review_queue(run: RunResult) -> list[dict[str, Any]]:
    """Everything a person still has to decide: pending fixes and unreviewed anomalies."""
    out = [{"type": "suggestion", "id": s.id, "sheet": s.sheet, "row": s.xl_row, "column": s.column,
            "cell_ref": s.cell_ref, "action": s.action, "old": s.old, "new": s.new, "method": s.method,
            "safety": s.safety, "confidence": s.confidence, "rule_id": s.rule_id}
           for s in run.suggestions if s.status == "pending"]
    out += [{"type": "anomaly", "id": a.id, "sheet": a.sheet, "row": a.xl_row, "column": a.column,
             "cell_ref": a.cell_ref, "value": a.value, "detector": a.detector, "message": a.message}
            for a in run.anomalies if a.status == "open"]
    return out


def removed_rows(run: RunResult, exported: ExportResult) -> list[dict[str, Any]]:
    issues = {i.id: i for i in run.validation.issues}
    out = []
    for c in exported.apply.applied:
        if c["action"] != "delete_row":
            continue
        sheet, row = c.get("sheet"), c.get("row")
        df = run.wb.sheets[sheet].df if sheet in run.wb.sheets else None
        values = ({str(k): jsonable(v) for k, v in df.loc[row].items()}
                  if df is not None and row in df.index else None)
        issue = issues.get(c["issue_id"])
        out.append({"sheet": sheet, "row_original": row, "reason": issue.message if issue else c.get("old"),
                    "duplicate_of": (issue.context or {}).get("duplicate_of") if issue else None,
                    "exact_copy": bool((issue.context or {}).get("exact")) if issue else None,
                    "values": values, "approved_by": c.get("approved_by")})
    return out


def not_verified_columns(run: RunResult) -> list[dict[str, str]]:
    """Free-text columns whose *spelling* nothing checks: no allowed list, aliases, regex or format.

    Typos there (a city written 'Hydrabad') cannot be detected without a reference, so the report
    says so instead of implying the column was cleaned.
    """
    out = []
    for srules in run.rules.sheets:
        name = srules.sheet or run.wb.first_sheet.name
        prof = run.profiles.get(name)
        if prof is None:
            continue
        looked_up = {r.lookup.column for r in srules.rules if r.lookup}  # checked against another sheet
        for col, c in prof.columns.items():
            rule = srules.columns.get(col)
            if c.inferred_type not in ("text", "category") or c.is_identifier or c.semantic or col in looked_up:
                continue
            if rule and (rule.allowed or rule.aliases or rule.regex or rule.format):
                continue
            out.append({"sheet": name, "column": str(col)})
    return out


def write_reports(run: RunResult, exported: ExportResult | None, out_dir: Path,
                  audit: dict[str, Any] | None = None) -> dict[str, Path]:
    audit = audit or build_audit(run, exported)
    stem = Path(run.wb.filename).stem
    paths = {
        "json": out_dir / f"{stem}_cleaning_report.json",
        "xlsx": out_dir / f"{stem}_cleaning_report.xlsx",
        "html": out_dir / f"{stem}_cleaning_report.html",
    }
    write_json(audit, paths["json"])
    write_xlsx(audit, paths["xlsx"])
    write_html(audit, paths["html"])
    return paths


def write_json(audit: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------- Excel


def write_xlsx(audit: dict[str, Any], path: Path) -> None:
    s = audit["summary"]
    after = audit.get("after") or {}
    if audit.get("output"):
        left_e, left_w = after.get("errors", 0), after.get("warnings", 0)
        state = (f"NOT FULLY CLEAN: {left_e} error(s) and {left_w} warning(s) remain; see the Unresolved sheet"
                 if left_e or left_w else "Clean: no rule violations remain")
        partial = (audit.get("source") or {}).get("partial")
        if partial:
            state = f"PARTIAL IMPORT ({partial_text(partial)}). {state} in the imported part"
    else:
        state = "Validated, not exported yet"
    resolutions = s.get("resolutions", {})
    not_verified = ", ".join(f"{c['sheet']}.{c['column']}" for c in audit.get("not_verified", []))
    summary_rows = [
        ("Result", state),
        ("Mode", "Safe Auto-Clean" if audit["config"]["mode"] == "safe_auto_clean" else "Manual review"),
        ("Items needing review", len(audit.get("review_required", []))),
        ("Rows removed", len(audit.get("removed_rows", []))),
        ("Tool", f"{audit['tool']['name']} {audit['tool']['version']}"),
        ("Run id", audit["run"]["id"]),
        ("Validated at", audit["run"]["created_at"]),
        ("Exported at", audit["run"]["exported_at"]),
        ("Input file", audit["input"]["filename"]),
        ("Input SHA-256", audit["input"]["sha256"]),
        ("Rules SHA-256", audit["rules"]["sha256"]),
        ("Output file", (audit.get("output") or {}).get("filename")),
        ("Output SHA-256", (audit.get("output") or {}).get("sha256")),
        *[(f"Rows checked: {name}", n) for name, n in s["rows_by_sheet"].items()],
        ("Quality score formula", s["quality_formula"]),
        ("Quality score (before)", s["quality_calculation"]),
        ("Quality score (after, from re-validating the corrected file)", after.get("quality_calculation")),
        ("Errors (before)", s["errors"]),
        ("Errors unresolved (after)", after.get("errors")),
        ("Warnings (before)", s["warnings"]),
        ("Warnings unresolved (after)", after.get("warnings")),
        ("Anomalies", s["anomalies"]),
        *[(f"Issues {k}", resolutions.get(k, 0)) for k in RESOLUTIONS if resolutions.get(k)],
        ("Changes applied", len(audit["changes"])),
        ("Changes skipped", len(audit["skipped_changes"])),
        ("Suggestions still pending (not applied)", s["suggestions_by_status"]["pending"]),
        ("Suggestions rejected", s["suggestions_by_status"]["rejected"]),
        ("Spelling not verified (no reference list)", not_verified or "-"),
        ("Source format", (audit.get("source") or {}).get("format")),
        ("Post-export verification", _verification_text(audit.get("verification"))),
    ]
    src = audit.get("source") or {}
    source_rows = [{"section": "table", "name": n, "detail": m.get("source", ""),
                    "notes": "; ".join(m.get("notes", []))} for n, m in src.get("tables", {}).items()]
    source_rows += [{"section": "excluded (not imported)", "name": record_where(p), "detail": p["problem"],
                     "notes": p.get("raw", "")} for p in src.get("excluded_records", [])]
    source_rows += [{"section": "warning", "name": "", "detail": w, "notes": ""} for w in src.get("warnings", [])]
    source_rows += [{"section": "not converted", "name": u.get("path", ""), "detail": u.get("reason", ""),
                     "notes": ""} for u in src.get("unsupported", [])]
    source_rows += [{"section": "documentation", "name": d.get("path", ""), "detail": d.get("kind", ""),
                     "notes": "missing-value codes mentioned: " + ", ".join(str(h.get("token")) for h in
                                                                         d.get("missing_value_hints", []))}
                    for d in src.get("documentation", [])]
    source_rows += [{"section": "structure", "name": "  " * n.get("depth", 0) + str(n.get("path")),
                     "detail": f"{n.get('kind')} {n.get('shape')}",
                     "notes": json.dumps(n["attrs"], ensure_ascii=False) if n.get("attrs") else ""}
                    for n in src.get("structure", [])]
    profile_rows = []
    for sheet in audit["profiles"]:
        for c in sheet["columns"]:
            num = c.get("numeric") or {}
            profile_rows.append({
                "sheet": sheet["name"], "column": c["name"], "type": c["inferred_type"],
                "type_share": c["type_share"], "rows": c["n_rows"], "missing": c["n_missing"],
                "missing_rate": c["missing_rate"], "distinct": c["n_distinct"],
                "min": num.get("min", (c.get("dates") or {}).get("min")),
                "max": num.get("max", (c.get("dates") or {}).get("max")),
                "mean": num.get("mean"), "whitespace": c["n_whitespace"],
                "numbers_as_text": c["n_numeric_text"], "dates_as_text": c["n_date_text"],
                "top_values": ", ".join(f"{t['value']} ({t['count']})" for t in c["top_values"][:5]),
            })
    issue_cols = ["id", "sheet", "cell_ref", "column", "rule_id", "severity", "value", "message", "resolution",
                  "resolution_note", "why"]
    anomaly_cols = ["id", "sheet", "cell_ref", "column", "detector", "score", "value", "message", "status",
                    "resolution"]
    unresolved_cols = ["corrected_cell_ref", "original_cell_ref", "rule_id", "severity", "value", "message",
                       "resolution", "resolution_note", "introduced_by_fix"]
    change_cols = ["suggestion_id", "cell_ref", "action", "old", "new", "method", "safety", "rule_id",
                   "approved_by", "approved_at"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(summary_rows, columns=["item", "value"]).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(audit.get("findings", []), columns=["category", "finding", "count", "where", "detail"]).to_excel(
            xw, sheet_name="Findings", index=False)
        pd.DataFrame(source_rows, columns=["section", "name", "detail", "notes"]).to_excel(
            xw, sheet_name="Source & structure", index=False)
        pd.DataFrame(audit.get("unresolved", []), columns=unresolved_cols).to_excel(
            xw, sheet_name="Unresolved", index=False)
        review_cols = ["type", "cell_ref", "action", "old", "new", "method", "safety", "confidence", "rule_id",
                       "detector", "value", "message", "id"]
        pd.DataFrame(audit.get("review_required", []), columns=review_cols).to_excel(
            xw, sheet_name="Review required", index=False)
        pd.DataFrame(_ml_rows(audit), columns=["section", "sheet", "name", "detail", "value"]).to_excel(
            xw, sheet_name="Dataset for ML", index=False)
        pd.DataFrame(profile_rows).to_excel(xw, sheet_name="Profile", index=False)
        pd.DataFrame(audit["issues"], columns=issue_cols).to_excel(xw, sheet_name="Issues", index=False)
        pd.DataFrame(audit["anomalies"], columns=anomaly_cols).to_excel(xw, sheet_name="Anomalies", index=False)
        pd.DataFrame(audit["changes"], columns=[*change_cols[:2], "row_cleaned", *change_cols[2:]]).to_excel(
            xw, sheet_name="Changes", index=False)
        removed = [{**r, "values": json.dumps(r.get("values"), ensure_ascii=False, default=str)}
                   for r in audit.get("removed_rows", [])]
        pd.DataFrame(removed, columns=["sheet", "row_original", "reason", "duplicate_of", "exact_copy",
                                       "approved_by", "values"]).to_excel(xw, sheet_name="Removed rows", index=False)
        skipped = [{**c, "reason": c.get("reason")} for c in audit["skipped_changes"]]
        pd.DataFrame(skipped, columns=[*change_cols, "reason"]).to_excel(xw, sheet_name="Skipped", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))


def _verification_text(v: dict[str, Any] | None) -> str:
    if not v:
        return "not run (nothing exported yet)"
    failed = [c for c in v.get("checks", []) if not c["passed"]]
    if failed:
        return "FAILED: " + "; ".join(f"{c['check']}: {c['detail']}" for c in failed)
    return f"passed ({len(v.get('checks', []))} checks: files reopened, rows/columns match, fixes present, re-validated)"


def _capped(items, sep: str, limit: int = 50) -> str:
    """Join at most `limit` items for a spreadsheet cell; the JSON report keeps the full list."""
    items = list(items)
    text = sep.join(items[:limit])
    return text + (f"{sep}... ({len(items)} in total; full list in the JSON report)" if len(items) > limit else "")


def _ml_rows(audit: dict[str, Any]) -> list[tuple]:
    rows: list[tuple] = []
    for schema in audit.get("columns", []):
        sheet = schema["sheet"]
        for c in schema["columns"]:
            rows.append(("column", sheet, c["name"], f"role={c['role']}, status={c.get('status', 'ready')}",
                         c["dtype"]))
        for role, cols in schema["features"].items():
            if cols:
                rows.append(("feature list (ready)", sheet, role, ", ".join(cols), len(cols)))
        for r in schema.get("needs_review", []):
            rows.append(("needs review before use", sheet, r["column"], "; ".join(r["reasons"]), r["role"]))
        for kind in ("identifier", "pii", "empty", "constant"):
            if schema["excluded"][kind]:
                rows.append(("excluded from features", sheet, kind, ", ".join(schema["excluded"][kind]), ""))
        if schema["high_missing"]:
            rows.append(("over 50% missing (kept; decide in your pipeline)", sheet, "",
                         ", ".join(schema["high_missing"]), len(schema["high_missing"])))
    for entry in audit.get("ml", {}).get("sheets", []):
        if not entry.get("target"):
            continue
        sheet, ts = entry["sheet"], entry.get("target_summary") or {}
        rows.append(("target", sheet, entry["target"], ts.get("kind", ""), f"{ts.get('missing', 0)} missing"))
        if ts.get("valid_labels"):
            rows.append(("valid labels", sheet, entry["target"], ", ".join(ts["valid_labels"]), len(ts["valid_labels"])))
        for c in ts.get("classes", []):
            rows.append(("class balance", sheet, c["label"], f"{c['share']:.1%}", c["count"]))
        for u in ts.get("unrecognised", []):
            rows.append(("unrecognised label (not guessed)", sheet, u["label"], "", u["count"]))
        missing = entry.get("missing_target_rows", [])
        conflicts = entry.get("label_conflicts", [])
        rows.append(("missing labels (rows)", sheet, "", _capped(map(str, missing), ", "), len(missing)))
        rows.append(("label conflicts (row groups)", sheet, "",
                     _capped((" vs ".join(map(str, g)) for g in conflicts), "; "), len(conflicts)))
        for s in entry.get("leakage_suspects", []):
            rows.append(("possible leakage", sheet, s["column"], s["detail"], s["kind"]))
    for k in (audit.get("dataset") or {}).get("kept_as_text", []):
        rows.append(("kept as text", k["sheet"], k["column"], f"intended {k['intended_dtype']}",
                     f"{k['unparseable_values']} unparseable"))
    return rows


# ----------------------------------------------------------------------------- HTML


def _histogram_svg(hist: dict[str, list[float]] | None, w: int = 168, h: int = 40) -> str:
    if not hist or not hist.get("counts"):
        return ""
    counts, edges = hist["counts"], hist["edges"]
    peak = max(counts) or 1
    n = len(counts)
    gap = 2
    bw = (w - gap * (n - 1)) / n
    bars = []
    for i, c in enumerate(counts):
        bh = 0 if c == 0 else max(2, (h - 2) * c / peak)
        x = i * (bw + gap)
        tip = f"{edges[i]:,.4g} to {edges[i + 1]:,.4g}: {c} rows"
        bars.append(
            f'<g><title>{tip}</title><rect x="{x:.1f}" y="0" width="{bw:.1f}" height="{h}" fill="transparent"/>'
            f'<rect class="bar" x="{x:.1f}" y="{h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="1.5"/></g>'
        )
    svg = (f'<svg class="hist" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
           f'aria-label="Distribution histogram">{"".join(bars)}</svg>')
    outside = hist.get("outside", 0)
    if outside:
        svg += f'<div class="why">+{outside} outside the 1st–99th percentile</div>'
    return svg


def write_html(audit: dict[str, Any], path: Path) -> None:
    env = Environment(loader=PackageLoader("ilovecsv", "templates"), autoescape=select_autoescape(["html", "j2"]))
    env.filters["hist"] = _histogram_svg
    by_rule = audit["summary"]["by_rule"]
    peak = max(by_rule.values(), default=1)
    severity_of = {}
    for i in audit["issues"]:
        severity_of.setdefault(i["rule_id"], i["severity"])
    rule_bars = [
        {"rule": r, "count": c, "pct": 100 * c / peak, "severity": severity_of.get(r, "error")}
        for r, c in by_rule.items()
    ]
    html = env.get_template("report.html.j2").render(
        a=audit, rule_bars=rule_bars, issues=audit["issues"][:1000],
        anomalies=audit["anomalies"][:500], truncated=len(audit["issues"]) > 1000,
    )
    path.write_text(html, encoding="utf-8")
