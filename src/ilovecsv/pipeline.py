"""End-to-end orchestration used by the CLI, the API and the tests.

    run = run_checks("orders.xlsx", rules_yaml)       # profile + validate + anomalies + suggestions
    decide(run, approve=[...], approver="alice")       # human approval step
    exported = export(run, "out/")                     # corrected file + re-validation + reports
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .anomalies import detect_anomalies
from .corrections import ApplyResult, apply_corrections, mark_unresolved, suggest_corrections
from .models import ANOMALY, APPROVED, ERROR, PENDING, REJECTED, Issue, Suggestion
from .profiling import SheetProfile, profile_workbook
from .reader import WorkbookData, read_workbook, sha256_file
from .resolution import apply_pre_export, finalize
from .rules.engine import ValidationResult, validate
from .rules.schema import RuleSet, load_rules, rules_sha256
from .values import normalise_nulls


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunResult:
    run_id: str
    created_at: str
    wb: WorkbookData
    rules: RuleSet
    rules_text: str
    rules_sha256: str
    profiles: dict[str, SheetProfile]
    validation: ValidationResult
    anomalies: list[Issue]
    suggestions: list[Suggestion]

    @property
    def issues(self) -> list[Issue]:
        return self.validation.issues + self.anomalies

    def summary(self) -> dict[str, Any]:
        return summarise(self.wb, self.rules, self.validation.issues, self.anomalies,
                         self.suggestions, self.validation.rule_errors)


@dataclass
class ExportResult:
    corrected_path: Path
    apply: ApplyResult
    after: dict[str, Any]
    reports: dict[str, Path] = field(default_factory=dict)


def run_checks(
    path: str | Path,
    rules_text: str,
    filename: str | None = None,
    run_id: str | None = None,
) -> RunResult:
    rules = load_rules(rules_text)
    wb = read_workbook(path, filename)
    return run_on_workbook(wb, rules, rules_text, run_id)


def run_on_workbook(wb: WorkbookData, rules: RuleSet, rules_text: str, run_id: str | None = None) -> RunResult:
    nulls = normalise_nulls(rules.null_values)
    profiles = profile_workbook(wb, nulls)
    validation = validate(wb, rules, profiles)
    anomalies = detect_anomalies(wb, rules, profiles, validation.issues)
    suggestions = suggest_corrections(wb, rules, validation.issues, profiles)
    apply_pre_export(validation.issues + anomalies, suggestions)
    return RunResult(
        run_id=run_id or uuid.uuid4().hex[:12],
        created_at=now_iso(),
        wb=wb,
        rules=rules,
        rules_text=rules_text,
        rules_sha256=rules_sha256(rules_text),
        profiles=profiles,
        validation=validation,
        anomalies=anomalies,
        suggestions=suggestions,
    )


def decide(
    run: RunResult,
    approve: Iterable[str] = (),
    reject: Iterable[str] = (),
    approver: str = "cli",
    min_confidence: float | None = None,
    include_row_deletions: bool = False,
) -> int:
    """Approve/reject suggestions by id, or approve all at or above min_confidence.

    Bulk approval by confidence never deletes rows unless include_row_deletions is set:
    duplicate removal should be approved after looking at both rows.
    """
    approve, reject = set(approve), set(reject)
    changed = 0
    for s in run.suggestions:
        target = None
        bulk_ok = include_row_deletions or s.action != "delete_row"
        if s.id in reject:
            target = REJECTED
        elif s.id in approve or (min_confidence is not None and s.confidence >= min_confidence and bulk_ok):
            target = APPROVED
        if target and s.status != target:
            s.status, s.decided_by, s.decided_at = target, approver, now_iso()
            changed += 1
    apply_pre_export(run.issues, run.suggestions)
    return changed


def export(run: RunResult, out_dir: str | Path, annotate: bool = True, write_reports: bool = True) -> ExportResult:
    from .report import write_reports as _write_reports  # local import: jinja2 only needed here

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(run.wb.filename).stem
    suffix = ".csv" if run.wb.file_type == "csv" else ".xlsx"
    corrected = out_dir / f"{stem}_corrected{suffix}"
    applied = apply_corrections(run.wb, run.suggestions, corrected, annotate=annotate)

    # Re-validate the corrected file with the same rules: proves what the fixes achieved.
    # Read it under the original filename: a CSV's sheet name comes from its file name.
    wb2 = read_workbook(corrected, run.wb.filename)
    after_run = run_on_workbook(wb2, run.rules, run.rules_text, run_id=run.run_id + "-after")
    resolution = finalize(run, applied, after_run.validation.issues, wb2.sheets)
    if annotate and run.wb.file_type == "xlsx" and resolution.unresolved:
        mark_unresolved(corrected, resolution.unresolved)  # visible in the file, not only in the report
        applied.sha256 = sha256_file(corrected)
    after = after_run.summary()
    after["unresolved"] = resolution.unresolved
    after["resolution_counts"] = resolution.counts

    result = ExportResult(corrected, applied, after)
    if write_reports:
        result.reports = _write_reports(run, result, out_dir)
    return result


def summarise(
    wb: WorkbookData,
    rules: RuleSet,
    errors: list[Issue],
    anomalies: list[Issue],
    suggestions: list[Suggestion],
    rule_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    checked_sheets = [s.sheet or wb.first_sheet.name for s in rules.sheets]
    rows = sum(len(wb.sheets[s].df) for s in checked_sheets if s in wb.sheets)
    cells = sum(wb.sheets[s].df.size for s in checked_sheets if s in wb.sheets)
    rows_by_sheet = {s: len(wb.sheets[s].df) for s in checked_sheets if s in wb.sheets}

    # Quality score = 100 * (1 - (E + 0.5 * W) / C)
    #   E = cells with at least one error (a duplicate row counts all of its cells; anomalies a
    #       reviewer confirmed count as errors), W = cells with only warnings, C = cells checked.
    error_cells: set[tuple[str, int, str]] = set()
    warning_cells: set[tuple[str, int, str]] = set()
    for i in errors + [a for a in anomalies if a.status == "confirmed_error"]:
        if i.xl_row is None or i.sheet not in wb.sheets:
            continue
        cols = [i.column] if i.column else list(wb.sheets[i.sheet].df.columns)
        target = error_cells if (i.severity == "error" or i.kind == ANOMALY) else warning_cells
        target.update((i.sheet, i.xl_row, c) for c in cols)
    warning_cells -= error_cells
    penalty = len(error_cells) + 0.5 * len(warning_cells)

    by_rule = Counter(i.rule_id for i in errors)
    status = Counter(s.status for s in suggestions)
    score = max(0.0, math.floor(10000 * (1 - penalty / cells)) / 100) if cells else 100.0
    return {
        "sheets_checked": checked_sheets,
        "rows_by_sheet": rows_by_sheet,
        "rows": rows,
        "cells": cells,
        "errors": sum(1 for i in errors if i.kind == ERROR and i.severity == "error"),
        "warnings": sum(1 for i in errors if i.kind == ERROR and i.severity == "warning"),
        "anomalies": sum(1 for i in anomalies if i.kind == ANOMALY),
        "anomalies_open": sum(1 for i in anomalies if i.status == "open"),
        "anomalies_confirmed": sum(1 for i in anomalies if i.status == "confirmed_error"),
        "cells_with_errors": len(error_cells),
        "cells_with_warnings": len(warning_cells),
        # Floored, so a sheet with any issue never displays as 100%.
        "quality_score": score,
        "quality_formula": "100 x (1 - (cells with errors + 0.5 x cells with only warnings) / cells checked)",
        "quality_calculation": (f"100 x (1 - ({len(error_cells)} + 0.5 x {len(warning_cells)}) / {cells}) = {score}"
                                if cells else "no cells checked"),
        "resolutions": dict(Counter(i.resolution for i in errors + anomalies if i.resolution)),
        "by_rule": dict(by_rule.most_common()),
        "suggestions": len(suggestions),
        "suggestions_by_status": {k: status.get(k, 0) for k in (PENDING, APPROVED, REJECTED)},
        "rule_errors": rule_errors,
    }


def tool_info() -> dict[str, str]:
    return {"name": "IloveCSV", "version": __version__}
