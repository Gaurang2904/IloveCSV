"""Glue between the API, the database and the core pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..db import IssueRow, Run, SuggestionRow, User, Workbook, utcnow
from ..models import stable_id
from ..pipeline import ExportResult, RunResult, export, run_on_workbook
from ..profiling import profile_workbook
from ..reader import ReadError, read_workbook
from ..report import write_reports
from ..resolution import apply_pre_export
from ..rules.schema import RulesError, load_rules
from ..rules.suggest import suggest_rules, with_prepare_options
from ..sources import SourceOptions
from .errors import Problem, read_problem
from .settings import Settings


def load_wb(row: Workbook, source: SourceOptions | None = None):
    try:
        return read_workbook(row.stored_path, row.filename, source)
    except ReadError as e:
        raise read_problem(e) from e


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
    wb = load_wb(wb_row, rules.source)
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
    rules = parse_rules(run.rules_text)
    result = run_on_workbook(load_wb(wb_row, rules.source), rules, run.rules_text, run_id=run.id)
    result.created_at = run.created_at
    result.auto_cleaned = run.mode == "safe_auto_clean"
    issue_state = {r.id: r for r in db.scalars(select(IssueRow).where(IssueRow.run_id == run.id))}
    for i in result.issues:
        if i.id in issue_state:
            i.status = issue_state[i.id].status
    sug_state = {r.id: r for r in db.scalars(select(SuggestionRow).where(SuggestionRow.run_id == run.id))}
    for s in result.suggestions:
        row = sug_state.get(s.id)
        if row is not None:
            s.status, s.decided_by, s.decided_at = row.status, row.decided_by, row.decided_at
    apply_pre_export(result.issues, result.suggestions)
    return result


def refresh_resolutions(db: Session, run: Run) -> None:
    """Recompute pre-export resolutions after a decision (works directly on the DB rows)."""
    issues = list(db.scalars(select(IssueRow).where(IssueRow.run_id == run.id,
                                                    IssueRow.introduced_by_fix.is_not(True))))
    sugs = list(db.scalars(select(SuggestionRow).where(SuggestionRow.run_id == run.id)))
    apply_pre_export(issues, sugs)
    db.flush()


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
    summary["resolutions"] = dict(db.execute(
        select(IssueRow.resolution, func.count()).where(IssueRow.run_id == run.id,
                                                        IssueRow.introduced_by_fix.is_not(True))
        .group_by(IssueRow.resolution)).all())
    run.summary = summary


def approve_safe(db: Session, run: Run, approver: str) -> int:
    """Safe Auto-Clean on the stored suggestions: approve every pending 'safe' one."""
    now, n = utcnow(), 0
    for s in db.scalars(select(SuggestionRow).where(SuggestionRow.run_id == run.id, SuggestionRow.safety == "safe",
                                                    SuggestionRow.status == "pending")):
        s.status, s.decided_by, s.decided_at = "approved", approver, now
        n += 1
    run.mode = "safe_auto_clean"
    db.flush()
    return n


def prepare_run(db: Session, user: User, wb_row: Workbook, settings: Settings, target: str | None,
                remove_exact_duplicates: bool, formats: list[str], rules_yaml: str | None = None,
                target_labels: list[str] | None = None,
                source: SourceOptions | None = None) -> tuple[Run, ExportResult, int]:
    """One click: suggested (or saved) rules + options, validate, Safe Auto-Clean, export."""
    if rules_yaml:
        saved = parse_rules(rules_yaml).source
        source = saved.model_copy(update=source.model_dump(exclude_unset=True)) if source else saved
    wb = load_wb(wb_row, source)
    text = rules_yaml or suggest_rules(profile_workbook(wb), for_ml=True)
    columns = {name: [str(c) for c in s.df.columns] for name, s in wb.sheets.items()}
    try:
        text = with_prepare_options(text, columns, target, remove_exact_duplicates, target_labels, source=source)
    except ValueError as e:
        raise Problem(422, "invalid-target", str(e)) from e
    run, _ = create_run(db, user, wb_row, text)
    db.flush()
    applied = approve_safe(db, run, f"{user.name} (safe auto-clean)")
    refresh_resolutions(db, run)
    exported = export_run(db, run, settings, annotate=True, formats=formats)
    return run, exported, applied


def export_run(db: Session, run: Run, settings: Settings, annotate: bool,
               formats: list[str] | tuple[str, ...] = ("csv",)) -> ExportResult:
    result = rebuild(db, run)
    out_dir = settings.data_dir / "runs" / run.id
    exported = export(result, out_dir, annotate=annotate, formats=tuple(formats))
    run.status = "exported"
    run.exported_at = utcnow()
    run.out_dir = str(out_dir)
    run.corrected_path = str(exported.corrected_path)
    run.reports = {
        **{k: str(v) for k, v in exported.reports.items()},
        "datasets": {sheet: {fmt: str(out_dir / name) for fmt, name in files.items()}
                     for sheet, files in exported.datasets.items()},
        "config": str(exported.config_path) if exported.config_path else None,
        **{k: str(v) for k, v in (("manifest", exported.manifest_path), ("feature_schema", exported.schema_path),
                                  ("script", exported.script_path)) if v},
    }
    run.after = {k: v for k, v in exported.after.items() if k != "unresolved"}
    run.summary = result.summary()
    _store_final_resolutions(db, run, result, exported.after.get("unresolved", []))
    return exported


def _store_final_resolutions(db: Session, run: Run, result: RunResult, unresolved: list[dict]) -> None:
    db.execute(delete(IssueRow).where(IssueRow.run_id == run.id, IssueRow.introduced_by_fix.is_(True)))
    rows = {r.id: r for r in db.scalars(select(IssueRow).where(IssueRow.run_id == run.id))}
    for i in result.issues:
        row = rows.get(i.id)
        if row is not None:
            row.resolution, row.resolution_note, row.corrected_cell_ref = i.resolution, i.resolution_note, None
    for u in unresolved:
        if u.get("original_issue_id") and u["original_issue_id"] in rows:
            rows[u["original_issue_id"]].corrected_cell_ref = u["corrected_cell_ref"]
            continue
        # A problem that only appeared after the corrections (e.g. a fixed category triggers a rule).
        sheet, ref = u["sheet"], u.get("original_cell_ref")
        orig_row = int("".join(ch for ch in ref.split("!", 1)[1] if ch.isdigit())) if ref else None
        db.add(IssueRow(
            run_id=run.id, id=stable_id("n", sheet, orig_row, u.get("column"), u["rule_id"]), kind="error",
            sheet=sheet, xl_row=orig_row, column=u.get("column"), cell_ref=ref, rule_id=u["rule_id"],
            severity=u["severity"], value=u.get("value"), message=u["message"], status="open",
            resolution=u["resolution"], resolution_note=u["resolution_note"], introduced_by_fix=True,
            corrected_cell_ref=u["corrected_cell_ref"],
        ))
    db.flush()
    refresh_counts(db, run)


def preview_reports(db: Session, run: Run, settings: Settings) -> dict[str, Path]:
    """Reports for a run that has not been exported yet (no corrected file)."""
    out_dir = settings.data_dir / "runs" / run.id / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_reports(rebuild(db, run), None, out_dir)
