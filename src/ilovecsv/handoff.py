"""The ML handoff: verification of the exported files, the manifest, the feature schema, the
recipe with review decisions, and a script that reproduces the preparation.

Stops before feature engineering: nothing here splits, scales, encodes or imputes. The manifest
records split hints (group and timestamp columns) because battery, patient or time-ordered data
must not be split at random, but the split itself belongs to the user's pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from . import __version__
from .models import APPROVED, REJECTED, jsonable
from .reader import sha256_file
from .rules.suggest import RECIPE_HEADER
from .values import to_text

if TYPE_CHECKING:
    from .pipeline import ExportResult, RunResult

MANIFEST_VERSION = "1.0"


# ----------------------------------------------------------------------------- verification


def verify_outputs(run: RunResult, ex: ExportResult) -> dict[str, Any]:
    """Re-open the exported files and check them against what the run says happened."""
    checks: list[dict[str, Any]] = []
    out_dir = ex.out_dir or ex.corrected_path.parent
    after = ex.after_run
    removed: dict[str, int] = {}
    for c in ex.apply.applied:
        if c["action"] == "delete_row":
            removed[c["sheet"]] = removed.get(c["sheet"], 0) + 1

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for sheet, files in ex.datasets.items():
        expected_cols = [str(c) for c in run.wb.sheets[sheet].df.columns]
        expected_rows = len(run.wb.sheets[sheet].df) - removed.get(sheet, 0)
        for fmt, name in files.items():
            path = out_dir / name
            try:
                df = (pd.read_csv(path, dtype=str, keep_default_na=False) if fmt == "csv" else pd.read_parquet(path))
            except Exception as e:  # pragma: no cover - would mean a broken export
                check(f"{sheet}: {fmt} file reopens", False, f"{name} could not be read: {e}")
                continue
            check(f"{sheet}: {fmt} file reopens", True, f"{name}: {len(df):,} rows x {len(df.columns)} columns")
            check(f"{sheet}: {fmt} keeps every column", list(map(str, df.columns)) == expected_cols,
                  "same columns, same order" if list(map(str, df.columns)) == expected_cols else
                  f"expected {expected_cols}, found {list(df.columns)}")
            check(f"{sheet}: {fmt} row count", len(df) == expected_rows,
                  f"{len(df):,} rows = {len(run.wb.sheets[sheet].df):,} read - {removed.get(sheet, 0)} removed"
                  if len(df) == expected_rows else f"expected {expected_rows:,}, found {len(df):,}")

    # Every applied cell change is visible in the re-read (exported) data.
    missing = []
    if after is not None:
        for c in ex.apply.applied:
            if c["action"] != "set" or c.get("sheet") not in after.wb.sheets:
                continue
            row = ex.to_cleaned_row.get(c["sheet"], {}).get(c["row"])
            df = after.wb.sheets[c["sheet"]].df
            if row is None or row not in df.index or c["column"] not in df.columns:
                missing.append(c["cell_ref"])
                continue
            got, want = to_text(df.at[row, c["column"]], frozenset()), to_text(c["new"], frozenset())
            if (got or None) != (want or None) and not _same_number(got, want):
                missing.append(c["cell_ref"])
    check("applied fixes present in the exported data", not missing,
          f"all {sum(1 for c in ex.apply.applied if c['action'] == 'set')} cell changes found" if not missing
          else f"{len(missing)} change(s) not found: {missing[:5]}")
    check("remaining problems re-validated on the exported data", after is not None,
          f"{ex.after.get('errors', 0)} error(s), {ex.after.get('warnings', 0)} warning(s) after cleaning")
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def _same_number(a: str | None, b: str | None) -> bool:
    try:
        return a is not None and b is not None and float(a) == float(b)
    except ValueError:
        return False


# ----------------------------------------------------------------------------- recipe


def recipe_text(run: RunResult) -> str:
    """The rules as run, plus the review decisions taken, ready for `prepare --config`."""
    doc = yaml.safe_load(run.rules_text) or {}
    approve = sorted(s.id for s in run.suggestions if s.status == APPROVED and s.decided_by not in
                     ("safe-auto-clean",) and not str(s.decided_by or "").endswith("(safe auto-clean)"))
    reject = sorted(s.id for s in run.suggestions if s.status == REJECTED)
    anomalies = {a.id: a.status for a in run.anomalies if a.status != "open"}
    if approve or reject or anomalies:
        doc["decisions"] = {k: v for k, v in (("approve", approve), ("reject", reject), ("anomalies", anomalies)) if v}
    else:
        doc.pop("decisions", None)
    return RECIPE_HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=None, width=100)


def reproduce_script(run: RunResult, config_name: str, formats: tuple[str, ...]) -> str:
    return f'''"""Reproduce this dataset preparation with IloveCSV {__version__}.

    python prepare_dataset.py                       # the preserved original next to this script
    python prepare_dataset.py path/to/new_batch     # the same recipe on another file with this schema

It re-runs the recipe ({config_name}): the same source options, rules, Safe Auto-Clean policy and
recorded review decisions. It does not split, scale, encode or impute: those steps belong in your
training pipeline, fitted on the training split only.
"""

import sys
from pathlib import Path

from ilovecsv.pipeline import prepare

HERE = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "original" / {run.wb.filename!r}
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "reproduced"
result = prepare(source, out_dir, config=HERE / {config_name!r}, formats={tuple(formats)!r})
print("dataset:", result.dataset_path())
print("report:", result.report)
'''


# ----------------------------------------------------------------------------- manifest


def split_hints(run: RunResult) -> list[dict[str, Any]]:
    hints = []
    for srules in run.rules.sheets:
        name = srules.sheet or run.wb.first_sheet.name
        groups = [c for c, r in srules.roles.items() if r == "group"]
        times = [c for c, r in srules.roles.items() if r == "timestamp"]
        sheet = run.wb.sheets.get(name)
        if run.wb.source_format == "nasa_battery_mat" and sheet is not None and "battery" in sheet.df.columns:
            groups = groups or ["battery"]
            times = times or (["start_time"] if name == "cycles" else ["cycle_index"])
        if groups or times:
            hints.append({"sheet": name, "group_columns": groups, "time_columns": times,
                          "advice": "keep all rows of a group in one split" * bool(groups)
                          + ("; " if groups and times else "") + "split chronologically" * bool(times)})
    return hints


def build_manifest(run: RunResult, ex: ExportResult, audit: dict[str, Any], files: dict[str, Path]) -> dict[str, Any]:
    out_dir = ex.out_dir or ex.corrected_path.parent
    wb = run.wb
    return {
        "manifest_version": MANIFEST_VERSION,
        "tool": {"name": "IloveCSV", "version": __version__},
        "run": {"id": run.run_id, "created_at": run.created_at, "mode": audit["config"]["mode"]},
        "source": {
            "filename": wb.filename, "format": wb.source_format or wb.file_type, "sha256": wb.sha256,
            "preserved_copy": audit["input"].get("preserved_copy"),
            "options": run.rules.source.model_dump(exclude_defaults=True),
            "partial": wb.partial,
            "documentation": [{"path": d["path"], "kind": d["kind"],
                               "missing_value_hints": d.get("missing_value_hints", [])} for d in wb.docs],
            "unsupported": wb.unsupported,
            "warnings": wb.warnings,
        },
        "tables": [
            {"name": name, "source": wb.tables_meta.get(name, {}).get("source", name),
             "conversion_notes": wb.tables_meta.get(name, {}).get("notes", []),
             "rows_read": len(wb.sheets[name].df),
             "rows_exported": (audit.get("dataset") or {}).get("rows_after", {}).get(name),
             "columns": [str(c) for c in wb.sheets[name].df.columns],
             "files": {fmt: {"file": f, "sha256": sha256_file(out_dir / f)} for fmt, f in ex.datasets.get(name, {}).items()}}
            for name in ex.datasets
        ],
        "target": {(s.sheet or wb.first_sheet.name): {"column": s.target, "labels": s.target_labels}
                   for s in run.rules.sheets if s.target},
        "roles": {(s.sheet or wb.first_sheet.name): s.roles for s in run.rules.sheets if s.roles},
        "split_hints": split_hints(run),
        "transformations": {
            "applied": _count(c["method"] for c in ex.apply.applied),
            "skipped": len(ex.apply.skipped),
            "pending_review": len(audit.get("review_required", [])),
            "not_done_by_design": ["imputation", "scaling", "encoding", "outlier removal", "column dropping",
                                   "train/test split"],
        },
        "recipe_decisions": run.recipe_decisions,
        "verification": audit.get("verification"),
        "files": {k: {"file": p.name, "sha256": sha256_file(p)} for k, p in files.items() if p and p.exists()},
    }


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def write_json(obj: Any, path: Path) -> Path:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=jsonable), encoding="utf-8")
    return path
