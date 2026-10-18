from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ...db import AuditEvent, IssueRow, Run, SuggestionRow, User, Workbook, utcnow
from ...explain import explain_issue
from ...models import ANOMALY, Issue, jsonable
from ..deps import audit, current_user, get_db
from ..errors import Problem
from ..schemas import DecisionIn, ExportIn, IssuePatch, issue_out, suggestion_out
from ..service import (
    export_run,
    get_run,
    load_wb,
    preview_reports,
    refresh_counts,
    refresh_resolutions,
)

router = APIRouter(prefix="/v1/runs", tags=["runs"])

MEDIA = {
    "html": "text/html",
    "json": "application/json",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "parquet": "application/vnd.apache.parquet",
}


def run_out(r: Run) -> dict:
    base = f"/v1/runs/{r.id}"
    links = {"issues": f"{base}/issues", "suggestions": f"{base}/suggestions", "stats": f"{base}/stats",
             "duplicates": f"{base}/duplicates",
             "decisions": f"{base}/decisions", "export": f"{base}/export", "report": f"{base}/report?format=html"}
    if r.status == "exported":
        links["corrected"] = f"{base}/corrected"
        links["dataset"] = f"{base}/dataset?format=csv"
        links["config"] = f"{base}/config"
    return {"id": r.id, "workbook_id": r.workbook_id, "status": r.status, "mode": r.mode or "manual_review",
            "created_at": r.created_at, "exported_at": r.exported_at, "rules_sha256": r.rules_sha256,
            "summary": r.summary, "after": r.after, "links": links}


@router.get("/{run_id}", summary="Run status and summary")
def get_one(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return run_out(get_run(db, user, run_id))


@router.get("/{run_id}/rules", summary="The rule file this run used")
def rules(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    r = get_run(db, user, run_id)
    return {"rules_yaml": r.rules_text, "sha256": r.rules_sha256}


@router.get("/{run_id}/issues", summary="List issues with filters")
def list_issues(
    run_id: str,
    kind: Literal["error", "anomaly"] | None = None,
    severity: Literal["error", "warning", "info"] | None = None,
    rule_id: str | None = None,
    column: str | None = None,
    sheet: str | None = None,
    status: Literal["open", "confirmed_error", "dismissed"] | None = None,
    resolution: Literal["resolved", "open", "pending_review", "approved", "rejected", "dismissed"] | None = None,
    q: str | None = Query(default=None, max_length=200, description="Search message, cell and value"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    get_run(db, user, run_id)
    stmt = select(IssueRow).where(IssueRow.run_id == run_id)
    for col, val in ((IssueRow.kind, kind), (IssueRow.severity, severity), (IssueRow.rule_id, rule_id),
                     (IssueRow.column, column), (IssueRow.sheet, sheet), (IssueRow.status, status),
                     (IssueRow.resolution, resolution)):
        if val is not None:
            stmt = stmt.where(col == val)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(IssueRow.message.ilike(like), IssueRow.cell_ref.ilike(like)))
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = db.scalars(stmt.order_by(IssueRow.sheet, IssueRow.xl_row, IssueRow.column).limit(limit).offset(offset))
    return {"items": [issue_out(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/{run_id}/stats", summary="Issue counts grouped by kind, rule and severity (SQL aggregate)")
def stats(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    get_run(db, user, run_id)
    rows = db.execute(
        select(IssueRow.kind, IssueRow.rule_id, IssueRow.severity, IssueRow.status, func.count().label("n"))
        .where(IssueRow.run_id == run_id)
        .group_by(IssueRow.kind, IssueRow.rule_id, IssueRow.severity, IssueRow.status)
        .order_by(func.count().desc())
    ).all()
    by_column = db.execute(
        select(IssueRow.sheet, IssueRow.column, func.count().label("n"))
        .where(IssueRow.run_id == run_id, IssueRow.kind == "error")
        .group_by(IssueRow.sheet, IssueRow.column).order_by(func.count().desc())
    ).all()
    return {
        "by_rule": [dict(r._mapping) for r in rows],
        "by_column": [dict(r._mapping) for r in by_column],
    }


@router.get("/{run_id}/issues/{issue_id}", summary="One issue with its suggestion and the full row")
def get_issue(run_id: str, issue_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    row = db.get(IssueRow, (run_id, issue_id))
    if row is None:
        raise Problem(404, "not-found", f"Issue '{issue_id}' was not found in run '{run_id}'.")
    sugs = db.scalars(select(SuggestionRow).where(SuggestionRow.run_id == run_id, SuggestionRow.issue_id == issue_id))
    context = None
    if row.xl_row is not None:
        wb = load_wb(db.get(Workbook, run.workbook_id))
        sheet = wb.sheets.get(row.sheet)
        if sheet is not None and row.xl_row in sheet.df.index:
            context = {c: jsonable(v) for c, v in sheet.df.loc[row.xl_row].items()}
    out = issue_out(row)
    out["why"] = explain_issue(Issue(row.kind, row.sheet, row.rule_id, row.severity, row.message))
    out["suggestions"] = [suggestion_out(s) for s in sugs]
    out["row"] = context
    return out


@router.patch("/{run_id}/issues/{issue_id}", summary="Confirm or dismiss an anomaly")
def patch_issue(run_id: str, issue_id: str, body: IssuePatch, user: User = Depends(current_user),
                db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    row = db.get(IssueRow, (run_id, issue_id))
    if row is None:
        raise Problem(404, "not-found", f"Issue '{issue_id}' was not found in run '{run_id}'.")
    if row.kind != ANOMALY:
        raise Problem(409, "not-an-anomaly",
                      "Rule errors are confirmed by the rule file and cannot be dismissed; change the rule instead.")
    before = row.status
    row.status, row.note = body.status, body.note
    row.decided_by, row.decided_at = user.name, utcnow()
    db.flush()
    _decisions_changed(db, run)
    audit(db, user, "issue.status_changed", run_id=run_id,
          detail={"issue_id": issue_id, "from": before, "to": body.status, "note": body.note})
    db.commit()
    return issue_out(row)


@router.get("/{run_id}/suggestions", summary="Suggested corrections")
def list_suggestions(
    run_id: str,
    status: Literal["pending", "approved", "rejected"] | None = None,
    method: str | None = None,
    rule_id: str | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    get_run(db, user, run_id)
    stmt = select(SuggestionRow).where(SuggestionRow.run_id == run_id)
    for col, val in ((SuggestionRow.status, status), (SuggestionRow.method, method), (SuggestionRow.rule_id, rule_id)):
        if val is not None:
            stmt = stmt.where(col == val)
    rows = list(db.scalars(stmt.order_by(SuggestionRow.sheet, SuggestionRow.xl_row)))
    return {"items": [suggestion_out(r) for r in rows], "total": len(rows)}


@router.post("/{run_id}/decisions", summary="Approve or reject suggestions")
def decide(run_id: str, body: DecisionIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    rows = {r.id: r for r in db.scalars(select(SuggestionRow).where(SuggestionRow.run_id == run_id))}
    unknown = sorted((set(body.approve) | set(body.reject)) - rows.keys())
    if unknown:
        raise Problem(422, "unknown-suggestions", "Some suggestion ids do not belong to this run.",
                      [{"loc": ["body"], "msg": f"unknown id {u}"} for u in unknown[:20]])
    both = set(body.approve) & set(body.reject)
    if both:
        raise Problem(422, "conflicting-decision", f"Ids both approved and rejected: {sorted(both)[:10]}")
    targets: dict[str, str] = {i: "approved" for i in body.approve}
    targets.update({i: "rejected" for i in body.reject})
    if body.approve_safe:
        for r in rows.values():
            if r.status == "pending" and r.safety == "safe" and r.id not in targets:
                targets[r.id] = "approved"
    if body.approve_min_confidence is not None:
        for r in rows.values():
            if (r.status == "pending" and r.id not in targets and r.confidence >= body.approve_min_confidence
                    and (body.include_row_deletions or r.action != "delete_row")
                    and (body.rule_id is None or r.rule_id == body.rule_id)
                    and (body.method is None or r.method == body.method)):
                targets[r.id] = "approved"
    now = utcnow()
    changed = []
    for sid, status in targets.items():
        r = rows[sid]
        if r.status != status:
            changed.append({"id": sid, "from": r.status, "to": status})
            r.status, r.decided_by, r.decided_at = status, user.name, now
    db.flush()
    if changed:
        _decisions_changed(db, run)
    else:
        refresh_counts(db, run)
    if changed:
        audit(db, user, "suggestions.decided", run_id=run_id, detail={"changes": changed})
    db.commit()
    return {"changed": len(changed), "suggestions_by_status": run.summary["suggestions_by_status"]}


def _decisions_changed(db: Session, run: Run) -> None:
    """A decision invalidates a previous export: resolutions go back to the review state."""
    if run.status == "exported":
        run.status = "validated"  # the corrected file no longer matches the decisions; export again
    refresh_resolutions(db, run)
    refresh_counts(db, run)


@router.get("/{run_id}/duplicates", summary="Preview duplicate rows before approving their removal")
def duplicates(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    issues = list(db.scalars(select(IssueRow).where(IssueRow.run_id == run_id, IssueRow.rule_id == "row.duplicate")))
    if not issues:
        return {"items": []}
    sugs = {s.issue_id: s for s in db.scalars(select(SuggestionRow).where(
        SuggestionRow.run_id == run_id, SuggestionRow.action == "delete_row"))}
    wb = load_wb(db.get(Workbook, run.workbook_id))
    items = []
    for i in issues:
        ctx = i.context or {}
        sheet = wb.sheets.get(i.sheet)
        first = ctx.get("duplicate_of")

        def row_values(r: int | None, sheet=sheet) -> dict | None:
            if sheet is None or r is None or r not in sheet.df.index:
                return None
            return {str(c): jsonable(v) for c, v in sheet.df.loc[r].items()}

        s = sugs.get(i.id)
        items.append({
            "issue_id": i.id, "sheet": i.sheet, "row": i.xl_row, "duplicate_of": first,
            "exact": bool(ctx.get("exact")), "compared_columns": ctx.get("compared_columns"),
            "ignored_columns": ctx.get("ignored_columns"), "match": ctx.get("match"),
            "row_values": row_values(i.xl_row), "duplicate_of_values": row_values(first),
            "suggestion_id": s.id if s else None, "suggestion_status": s.status if s else None,
            "confidence": s.confidence if s else None,
        })
    return {"items": items}


@router.post("/{run_id}/export", summary="Write the corrected workbook and audit reports")
def do_export(run_id: str, request: Request, body: ExportIn | None = None,
              user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    body = body or ExportIn()
    exported = export_run(db, run, request.app.state.settings, annotate=body.annotate, formats=list(body.formats))
    audit(db, user, "run.exported", run_id=run_id, detail={
        "applied": len(exported.apply.applied), "skipped": len(exported.apply.skipped),
        "output_sha256": exported.apply.sha256,
        "errors_before": run.summary["errors"], "errors_after": exported.after["errors"],
    })
    db.commit()
    out = run_out(run)
    out["applied"] = exported.apply.applied
    out["skipped"] = exported.apply.skipped
    pending = run.summary["suggestions_by_status"]["pending"]
    notices = []
    if not exported.apply.applied:
        notices.append(f"No changes were applied: the corrected file is an unchanged copy. "
                       f"{pending} suggestion(s) are still pending; approve them via /decisions first.")
    left_e, left_w = exported.after["errors"], exported.after["warnings"]
    if left_e or left_w:
        notices.append(f"Not fully clean: {left_e} error(s) and {left_w} warning(s) remain. They are listed "
                       "with resolution open/pending_review/rejected and highlighted in the corrected workbook.")
    if notices:
        out["notice"] = " ".join(notices)
    return out


@router.get("/{run_id}/report", summary="Download the audit report (html, xlsx or json)")
def report(run_id: str, request: Request, format: Literal["html", "xlsx", "json"] = "html",
           user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    if run.reports:
        path = Path(run.reports[format])
    else:
        path = preview_reports(db, run, request.app.state.settings)[format]
    audit(db, user, "report.downloaded", run_id=run_id, detail={"format": format, "exported": bool(run.reports)})
    db.commit()
    return FileResponse(path, media_type=MEDIA[format], filename=path.name)


@router.get("/{run_id}/corrected", summary="Download the corrected workbook")
def corrected(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    if not run.corrected_path:
        raise Problem(409, "not-exported", "Export the run first: POST /v1/runs/{run_id}/export.")
    path = Path(run.corrected_path)
    audit(db, user, "corrected.downloaded", run_id=run_id)
    db.commit()
    media = {".csv": MEDIA["csv"], ".zip": "application/zip"}.get(path.suffix, MEDIA["xlsx"])
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/{run_id}/dataset", summary="Download the cleaned dataset (CSV or Parquet)")
def dataset(run_id: str, format: Literal["csv", "parquet"] = "csv", sheet: str | None = None,
            user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    files = (run.reports or {}).get("datasets") or {}
    if not files:
        raise Problem(409, "not-exported", "Export or prepare the run first.")
    name = sheet or next(iter(files))
    if name not in files:
        raise Problem(404, "not-found", f"No dataset for sheet '{name}'. Available: {sorted(files)}.")
    if format not in files[name]:
        raise Problem(404, "not-found", f"No {format} file was written. Export with formats=['{format}'].")
    path = Path(files[name][format])
    audit(db, user, "dataset.downloaded", run_id=run_id, detail={"sheet": name, "format": format})
    db.commit()
    return FileResponse(path, media_type=MEDIA.get(format, "application/octet-stream"), filename=path.name)


@router.get("/{run_id}/config", summary="Download the prepare config (rules + policy) to reuse on new batches")
def config(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    path = (run.reports or {}).get("config")
    if not path:
        raise Problem(409, "not-exported", "Export or prepare the run first.")
    return FileResponse(Path(path), media_type="application/yaml", filename=Path(path).name)


HANDOFF = {"manifest": "application/json", "feature_schema": "application/json", "script": "text/x-python"}


@router.get("/{run_id}/handoff/{kind}", summary="Download a handoff file: manifest, feature_schema or script")
def handoff(run_id: str, kind: Literal["manifest", "feature_schema", "script"], user: User = Depends(current_user),
            db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    path = (run.reports or {}).get(kind)
    if not path:
        raise Problem(409, "not-exported", "Export or prepare the run first.")
    return FileResponse(Path(path), media_type=HANDOFF[kind], filename=Path(path).name)


@router.get("/{run_id}/audit-log", summary="Every action recorded for this run")
def audit_log(run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = get_run(db, user, run_id)
    rows = db.scalars(
        select(AuditEvent).where(or_(AuditEvent.run_id == run_id, AuditEvent.workbook_id == run.workbook_id))
        .order_by(AuditEvent.id)
    )
    names = dict(db.execute(select(User.id, User.name)).all())
    return {"items": [{"at": e.at, "user": names.get(e.user_id), "action": e.action, "run_id": e.run_id,
                       "workbook_id": e.workbook_id, "detail": e.detail} for e in rows]}
