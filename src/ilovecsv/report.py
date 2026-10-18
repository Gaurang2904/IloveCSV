"""Audit report: one data structure, three renderings (JSON, Excel, HTML).

Every report carries provenance: SHA-256 of the input file, the rule file and the
corrected output, the tool version, timestamps, and who approved each change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from .explain import explain_issue
from .models import ANOMALY, ERROR, RESOLUTIONS
from .pipeline import now_iso, tool_info

if TYPE_CHECKING:
    from .pipeline import ExportResult, RunResult


def build_audit(run: RunResult, exported: ExportResult | None = None) -> dict[str, Any]:
    errors = [i for i in run.validation.issues if i.kind == ERROR]
    anomalies = [i for i in run.anomalies if i.kind == ANOMALY]
    audit: dict[str, Any] = {
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
    }
    if exported:
        audit["changes"] = exported.apply.applied
        audit["skipped_changes"] = exported.apply.skipped
        audit["after"] = {k: v for k, v in exported.after.items() if k != "unresolved"}
        audit["unresolved"] = exported.after.get("unresolved", [])
        audit["output"] = {"filename": exported.corrected_path.name, "sha256": exported.apply.sha256}
    return audit


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


def write_reports(run: RunResult, exported: ExportResult | None, out_dir: Path) -> dict[str, Path]:
    audit = build_audit(run, exported)
    stem = Path(run.wb.filename).stem
    paths = {
        "json": out_dir / f"{stem}_audit.json",
        "xlsx": out_dir / f"{stem}_audit.xlsx",
        "html": out_dir / f"{stem}_report.html",
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
    else:
        state = "Validated, not exported yet"
    resolutions = s.get("resolutions", {})
    not_verified = ", ".join(f"{c['sheet']}.{c['column']}" for c in audit.get("not_verified", []))
    summary_rows = [
        ("Result", state),
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
    ]
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
    change_cols = ["suggestion_id", "cell_ref", "action", "old", "new", "method", "rule_id", "approved_by", "approved_at"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(summary_rows, columns=["item", "value"]).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(audit.get("unresolved", []), columns=unresolved_cols).to_excel(
            xw, sheet_name="Unresolved", index=False)
        pd.DataFrame(profile_rows).to_excel(xw, sheet_name="Profile", index=False)
        pd.DataFrame(audit["issues"], columns=issue_cols).to_excel(xw, sheet_name="Issues", index=False)
        pd.DataFrame(audit["anomalies"], columns=anomaly_cols).to_excel(xw, sheet_name="Anomalies", index=False)
        pd.DataFrame(audit["changes"], columns=change_cols).to_excel(xw, sheet_name="Changes", index=False)
        skipped = [{**c, "reason": c.get("reason")} for c in audit["skipped_changes"]]
        pd.DataFrame(skipped, columns=[*change_cols, "reason"]).to_excel(xw, sheet_name="Skipped", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))


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
