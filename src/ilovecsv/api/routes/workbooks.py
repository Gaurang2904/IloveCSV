from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...adapters import inspect_file
from ...db import User, Workbook
from ...profiling import profile_workbook
from ...reader import ReadError, sha256_file
from ...rules.suggest import suggest_rules
from ...sources import SourceOptions
from ..deps import audit, current_user, get_db
from ..errors import read_problem
from ..schemas import PrepareIn, RulesIn
from ..service import create_run, get_workbook, load_wb, parse_rules, prepare_run
from ..uploads import save_upload

router = APIRouter(prefix="/v1", tags=["workbooks"])


def workbook_out(w: Workbook) -> dict:
    return {
        "id": w.id, "filename": w.filename, "file_type": w.file_type, "sha256": w.sha256,
        "size_bytes": w.size_bytes, "uploaded_at": w.uploaded_at, "sheets": w.sheets, "warnings": w.warnings,
        "links": {"inspect": f"/v1/workbooks/{w.id}/inspect", "profile": f"/v1/workbooks/{w.id}/profile",
                  "suggested_rules": f"/v1/workbooks/{w.id}/rules/suggested",
                  "runs": f"/v1/workbooks/{w.id}/runs"},
    }


def _file_type(ext: str) -> str:
    if ext in (".xlsx", ".xlsm"):
        return "xlsx"
    return "csv" if ext in (".csv", ".tsv", ".txt") else "frames"


@router.post("/workbooks", status_code=201,
             summary="Upload a dataset (CSV/TSV, Excel, JSON/JSONL, Parquet, Feather, ARFF, ZIP, MAT, NPY/NPZ, HDF5)")
async def upload(request: Request, file: UploadFile = File(...), user: User = Depends(current_user),
                 db: Session = Depends(get_db)):
    settings = request.app.state.settings
    path, size, ext = await save_upload(file, settings)
    try:
        insp = inspect_file(path, file.filename)  # structure only: cheap even for large files
    except ReadError as e:
        path.unlink(missing_ok=True)
        raise read_problem(e) from e
    row = Workbook(
        id=uuid.uuid4().hex, owner_id=user.id, filename=(file.filename or "upload")[:255], stored_path=str(path),
        file_type=_file_type(ext), sha256=sha256_file(path), size_bytes=size, warnings=insp.warnings,
        sheets=[{"name": t.name, "rows": t.rows, "rows_exact": t.rows_exact,
                 "columns": [c["name"] for c in t.columns], "header_row": 1} for t in insp.tables],
    )
    db.add(row)
    audit(db, user, "workbook.uploaded", workbook_id=row.id,
          detail={"filename": row.filename, "sha256": row.sha256, "size_bytes": size})
    db.commit()
    return workbook_out(row)


@router.get("/workbooks", summary="List your workbooks")
def list_workbooks(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Workbook).where(Workbook.owner_id == user.id).order_by(Workbook.uploaded_at.desc()))
    return {"items": [workbook_out(w) for w in rows]}


@router.get("/workbooks/{workbook_id}", summary="Workbook metadata")
def get_one(workbook_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return workbook_out(get_workbook(db, user, workbook_id))


@router.get("/workbooks/{workbook_id}/inspect", summary="Structure first: tables, rows, columns, samples, docs")
def inspect(workbook_id: str, sheets: str | None = None, tables: str | None = None, json_records: str | None = None,
            mat_layout: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = get_workbook(db, user, workbook_id)
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    try:
        options = SourceOptions(sheets=split(sheets), tables=split(tables), json_records=json_records,
                                **({"mat_layout": mat_layout} if mat_layout else {}))
        insp = inspect_file(row.stored_path, row.filename, options)
    except (ReadError, ValueError) as e:
        raise read_problem(e) from e
    return {"workbook_id": workbook_id, **insp.to_dict()}


@router.get("/workbooks/{workbook_id}/profile", summary="Column profiles for every sheet")
def profile(workbook_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    wb = load_wb(get_workbook(db, user, workbook_id))
    return {"workbook_id": workbook_id, "sheets": [p.to_dict() for p in profile_workbook(wb).values()]}


@router.get("/workbooks/{workbook_id}/rules/suggested", summary="Draft rule file generated from the profile")
def suggested_rules(workbook_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    wb = load_wb(get_workbook(db, user, workbook_id))
    return {"rules_yaml": suggest_rules(profile_workbook(wb))}


@router.post("/rules/validate", summary="Check a rule file without running it")
def validate_rules(body: RulesIn, user: User = Depends(current_user)):
    rules = parse_rules(body.rules_yaml or "")
    return {"valid": True, "sheets": [
        {"sheet": s.sheet, "columns": list(s.columns), "rules": [r.id for r in s.rules]} for s in rules.sheets
    ]}


@router.post("/workbooks/{workbook_id}/runs", status_code=201, summary="Validate a workbook against a rule file")
def start_run(workbook_id: str, body: RulesIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    wb_row = get_workbook(db, user, workbook_id)
    text = body.rules_yaml
    if not text:
        text = suggest_rules(profile_workbook(load_wb(wb_row)))
    run, _ = create_run(db, user, wb_row, text)
    audit(db, user, "run.created", workbook_id=wb_row.id, run_id=run.id,
          detail={"rules_sha256": run.rules_sha256, "errors": run.summary["errors"],
                  "warnings": run.summary["warnings"], "anomalies": run.summary["anomalies"]})
    db.commit()
    from .runs import run_out

    return run_out(run)


@router.post("/workbooks/{workbook_id}/prepare", status_code=201,
             summary="One click: validate, apply Safe Auto-Clean and export the cleaned dataset")
def prepare(workbook_id: str, request: Request, body: PrepareIn | None = None, user: User = Depends(current_user),
            db: Session = Depends(get_db)):
    body = body or PrepareIn()
    wb_row = get_workbook(db, user, workbook_id)
    run, exported, applied = prepare_run(db, user, wb_row, request.app.state.settings, body.target,
                                         body.remove_exact_duplicates, list(body.formats), body.rules_yaml,
                                         body.target_labels, body.source)
    audit(db, user, "run.prepared", workbook_id=wb_row.id, run_id=run.id, detail={
        "target": body.target, "remove_exact_duplicates": body.remove_exact_duplicates,
        "safe_fixes_approved": applied, "applied": len(exported.apply.applied),
        "errors_after": exported.after["errors"], "warnings_after": exported.after["warnings"],
        "output_sha256": exported.apply.sha256,
    })
    db.commit()
    from .runs import run_out

    out = run_out(run)
    out["safe_fixes_applied"] = len(exported.apply.applied)
    out["review_required"] = run.summary["suggestions_by_status"]["pending"] + run.summary["anomalies_open"]
    out["datasets"] = exported.datasets
    return out
