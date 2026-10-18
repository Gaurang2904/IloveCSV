"""What happened to each issue: resolved, open, pending_review, approved, rejected or dismissed.

Before export the resolution follows the review decisions. After export it is decided by
re-validating the corrected file: an issue is *resolved* only if it no longer appears. Deleted
rows shift row numbers in the corrected file, so its issues are mapped back to the original
rows before comparing.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .models import (
    ANOMALY,
    APPROVED,
    CONFIRMED_ERROR,
    DISMISSED,
    DISMISSED_RES,
    ERROR,
    OPEN,
    PENDING,
    PENDING_REVIEW,
    REJECTED,
    RESOLVED,
    Issue,
    Suggestion,
)

if TYPE_CHECKING:
    from .corrections import ApplyResult
    from .pipeline import RunResult

_ANOMALY_RESOLUTION = {OPEN: PENDING_REVIEW, CONFIRMED_ERROR: OPEN, DISMISSED: DISMISSED_RES}
NO_FIX = "No safe automatic correction exists; fix the source data by hand."


def _suggestion_for(issue: Issue, by_issue: dict[str, Suggestion], by_cell: dict[str, Suggestion]) -> Suggestion | None:
    """The suggestion that addresses this issue: its own, or the one that won for the same cell."""
    return by_issue.get(issue.id) or (by_cell.get(issue.cell_ref) if issue.cell_ref and issue.column else None)


def _indexes(suggestions: list[Suggestion]) -> tuple[dict[str, Suggestion], dict[str, Suggestion]]:
    by_issue = {s.issue_id: s for s in suggestions}
    by_cell = {s.cell_ref: s for s in suggestions if s.action == "set" and s.cell_ref}
    return by_issue, by_cell


def apply_pre_export(issues: list[Issue], suggestions: list[Suggestion]) -> None:
    """Resolution from the review decisions alone (nothing has been written yet)."""
    by_issue, by_cell = _indexes(suggestions)
    for i in issues:
        if i.kind == ANOMALY:
            i.resolution, i.resolution_note = _ANOMALY_RESOLUTION.get(i.status, PENDING_REVIEW), None
            continue
        s = _suggestion_for(i, by_issue, by_cell)
        if s is None:
            i.resolution, i.resolution_note = OPEN, NO_FIX
        elif s.status == APPROVED:
            i.resolution, i.resolution_note = APPROVED, f"Fix approved ({s.method}); applied on export."
        elif s.status == REJECTED:
            i.resolution, i.resolution_note = REJECTED, f"Suggested fix ({s.method}) was rejected."
        else:
            i.resolution, i.resolution_note = PENDING_REVIEW, f"Suggested fix ({s.method}) awaits a decision."


@dataclass
class ResolutionReport:
    counts: dict[str, int]
    unresolved: list[dict[str, Any]] = field(default_factory=list)  # issues left in the corrected file


def row_maps(run: RunResult, apply: ApplyResult) -> tuple[dict[str, set[int]], dict[str, dict[int, int]]]:
    """Deleted original rows per sheet, and corrected row -> original row per sheet."""
    sug_by_id = {s.id: s for s in run.suggestions}
    deleted: dict[str, set[int]] = {name: set() for name in run.wb.sheets}
    for rec in apply.applied:
        s = sug_by_id.get(rec["suggestion_id"])
        if s is not None and s.action == "delete_row" and s.xl_row:
            deleted.setdefault(s.sheet, set()).add(s.xl_row)
    to_original: dict[str, dict[int, int]] = {}
    for name, sheet in run.wb.sheets.items():
        gone = sorted(deleted.get(name, set()))
        mapping: dict[int, int] = {}
        for r in range(1, (max(sheet.df.index) if len(sheet.df) else sheet.header_row) + 1):
            if r in deleted.get(name, set()):
                continue
            mapping[r - bisect_left(gone, r)] = r  # rows above r that were deleted shift it up
        to_original[name] = mapping
    return deleted, to_original


def finalize(run: RunResult, apply: ApplyResult, after_issues: list[Issue], after_sheets: dict) -> ResolutionReport:
    """Final resolution for every issue of `run`, decided by the re-validation of the corrected file."""
    deleted, to_original = row_maps(run, apply)
    by_issue, by_cell = _indexes(run.suggestions)
    skipped = {rec["suggestion_id"]: rec.get("reason", "") for rec in apply.skipped}
    applied_ids = {rec["suggestion_id"] for rec in apply.applied}

    def original_key(i: Issue) -> tuple:
        row = to_original.get(i.sheet, {}).get(i.xl_row, i.xl_row) if i.xl_row is not None else None
        return i.sheet, row, i.column, i.rule_id

    after_errors = [i for i in after_issues if i.kind == ERROR]
    remaining = {original_key(i) for i in after_errors}
    before_keys: dict[tuple, Issue] = {}

    for i in run.validation.issues:
        key = (i.sheet, i.xl_row, i.column, i.rule_id)
        before_keys[key] = i
        s = _suggestion_for(i, by_issue, by_cell)
        if i.xl_row is not None and i.xl_row in deleted.get(i.sheet, set()):
            i.resolution = RESOLVED
            i.resolution_note = ("Duplicate row removed." if i.rule_id == "row.duplicate"
                                 else "Row removed by an approved duplicate deletion.")
        elif key in remaining:
            if s is None:
                i.resolution, i.resolution_note = OPEN, NO_FIX
            elif s.status == REJECTED:
                i.resolution, i.resolution_note = REJECTED, f"Suggested fix ({s.method}) was rejected."
            elif s.status == PENDING:
                i.resolution, i.resolution_note = PENDING_REVIEW, f"Suggested fix ({s.method}) was not reviewed."
            else:
                reason = skipped.get(s.id) or "the fix was applied but the value still fails validation"
                i.resolution, i.resolution_note = OPEN, f"Approved fix not effective: {reason}."
        else:
            i.resolution = RESOLVED
            if s is not None and s.id in applied_ids:
                i.resolution_note = f"Fixed by {s.method}; passes re-validation."
            else:
                i.resolution_note = "Passes re-validation (resolved by another change)."

    for a in run.anomalies:
        a.resolution, a.resolution_note = _ANOMALY_RESOLUTION.get(a.status, PENDING_REVIEW), None

    unresolved: list[dict[str, Any]] = []
    for i in after_errors:
        sheet, orig_row, column, rule_id = original_key(i)
        before = before_keys.get((sheet, orig_row, column, rule_id))
        letter = after_sheets[sheet].col_letters.get(column) if column and sheet in after_sheets else None
        if orig_row is None:
            original_ref = None
        else:
            original_ref = f"{sheet}!{letter or 'A'}{orig_row}"
        entry = i.to_dict()
        entry.update({
            "corrected_cell_ref": i.cell_ref,
            "original_cell_ref": original_ref,
            "introduced_by_fix": before is None,
            "original_issue_id": before.id if before else None,
            "resolution": before.resolution if before else OPEN,
            "resolution_note": (before.resolution_note if before else
                                "Appeared after corrections were applied (for example, a corrected category "
                                "now triggers another rule)."),
        })
        unresolved.append(entry)

    counts = Counter(i.resolution for i in run.validation.issues + run.anomalies)
    return ResolutionReport(counts=dict(counts), unresolved=unresolved)
