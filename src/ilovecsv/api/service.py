"""Glue between the API, the database and the core pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import IssueRow, Run, SuggestionRow, User, Workbook, utcnow
from ..pipeline import ExportResult, RunResult, export, run_on_workbook
from ..reader import ReadError, read_workbook
from ..report import write_reports
from ..rules.schema import RulesError, load_rules
from .errors import Problem
from .settings import Settings


def load_wb(row: Workbook):
    try:
        return read_workbook(row.stored_path, row.filename)
    except ReadError as e:
        raise Problem(422, "unreadable-workbook", str(e)) from e


def parse_rules(text: str):
    try:
        return load_rules(text)
    except RulesError as e:
        raise Problem(422, "invalid-rules", str(e), e.errors) from e


def get_workbook(db: Session, user: User, workbook_id: str) -> Workbook:
    row = db.get(Workbook, workbook_id)
    # Someone else's workbook answers 404, not 403, so ids cannot be probed.
    if row is None or row.owner_id != user.id:
        raise Problem(404, "not-found", f"Workbook '{workbook_id}' was not found.")
    return row


def get_run(db: Session, user: User, run_id: str) -> Run:
    row = db.get(Run, run_id)
    if row is None or row.owner_id != user.id:
        raise Problem(404, "not-found", f"Run '{run_id}' was not found.")
    return row


def create_run(db: Session, user: User, wb_row: Workbook, rules_text: str) -> tuple[Run, RunResult]:
    rules = parse_rules(rules_text)
    wb = load_wb(wb_row)
    result = run_on_workbook(wb, rules, rules_text, run_id=uuid.uuid4().hex[:16])
    run = Run(id=result.run_id, workbook_id=wb_row.id, owner_id=user.id, rules_text=rules_text,
              rules_sha256=result.rules_sha256, created_at=result.created_at, summary=result.summary())
    db.add(run)
    db.flush()
    for i in result.issues:
        db.add(IssueRow(run_id=run.id, **{k: v for k, v in i.to_dict().items()}))
    for s in result.suggestions:
        db.add(SuggestionRow(run_id=run.id, **s.to_dict()))
    return run, result


def rebuild(db: Session, run: Run) -> RunResult:
    """Re-run the deterministic pipeline and overlay the decisions stored in the DB."""
    wb_row = db.get(Workbook, run.workbook_id)
    assert wb_row is not None
    result = run_on_workbook(load_wb(wb_row), parse_rules(run.rules_text), run.rules_text, run_id=run.id)
    result.created_at = run.created_at
    issue_state = {r.id: r for r in db.scalars(select(IssueRow).where(IssueRow.run_id == run.id))}
    for i in result.issues:
        if i.id in issue_state:
            i.status = issue_state[i.id].status
    sug_state = {r.id: r for r in db.scalars(select(SuggestionRow).where(SuggestionRow.run_id == run.id))}
    for s in result.suggestions:
        row = sug_state.get(s.id)
        if row is not None:
            s.status, s.decided_by, s.decided_at = row.status, row.decided_by, row.decided_at
    return result


def refresh_counts(db: Session, run: Run) -> None:
    """Update the decision counters in the run summary with SQL aggregates."""
    issue_status = dict(db.execute(
        select(IssueRow.status, func.count()).where(IssueRow.run_id == run.id, IssueRow.kind == "anomaly")
        .group_by(IssueRow.status)).all())
    sug_status = dict(db.execute(
        select(SuggestionRow.status, func.count()).where(SuggestionRow.run_id == run.id)
        .group_by(SuggestionRow.status)).all())
    summary = dict(run.summary)
    summary["anomalies_open"] = issue_status.get("open", 0)
    summary["anomalies_confirmed"] = issue_status.get("confirmed_error", 0)
    summary["suggestions_by_status"] = {k: sug_status.get(k, 0) for k in ("pending", "approved", "rejected")}
    run.summary = summary


def export_run(db: Session, run: Run, settings: Settings, annotate: bool) -> ExportResult:
    result = rebuild(db, run)
    out_dir = settings.data_dir / "runs" / run.id
    exported = export(result, out_dir, annotate=annotate)
    run.status = "exported"
    run.exported_at = utcnow()
    run.out_dir = str(out_dir)
    run.corrected_path = str(exported.corrected_path)
    run.reports = {k: str(v) for k, v in exported.reports.items()}
    run.after = {k: v for k, v in exported.after.items() if k != "remaining_errors"}
    run.summary = result.summary()
    return exported


def preview_reports(db: Session, run: Run, settings: Settings) -> dict[str, Path]:
    """Reports for a run that has not been exported yet (no corrected file)."""
    out_dir = settings.data_dir / "runs" / run.id / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_reports(rebuild(db, run), None, out_dir)
