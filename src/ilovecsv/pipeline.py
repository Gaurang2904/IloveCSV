"""End-to-end orchestration used by the CLI, the API and the tests.

    run = run_checks("orders.xlsx", rules_yaml)       # profile + validate + anomalies + suggestions
    decide(run, approve=[...], approver="alice")       # human approval step
    exported = export(run, "out/")                     # corrected file + re-validation + reports
"""

from __future__ import annotations

import math
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .anomalies import detect_anomalies
from .corrections import (
    ApplyResult,
    apply_corrections,
    apply_to_tables,
    cell_writeback,
    mark_unresolved,
    suggest_corrections,
)
from .models import ANOMALY, APPROVED, ERROR, PENDING, REJECTED, Issue, Suggestion
from .profiling import SheetProfile, profile_workbook
from .reader import WorkbookData, read_table_csvs, read_workbook, sha256_file
from .resolution import apply_pre_export, finalize, row_maps
from .rules.engine import ValidationResult, validate
from .rules.schema import RuleSet, load_rules, rules_sha256
from .rules.suggest import suggest_rules, with_prepare_options
from .sources import SourceOptions
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
    auto_cleaned: bool = False  # Safe Auto-Clean was applied (decide(mode="safe"))
    recipe_decisions: dict[str, int] | None = None  # decisions re-applied from a saved recipe

    @property
    def issues(self) -> list[Issue]:
        return self.validation.issues + self.anomalies

    def summary(self) -> dict[str, Any]:
        return summarise(self.wb, self.rules, self.validation.issues, self.anomalies,
                         self.suggestions, self.validation.rule_errors)


@dataclass
class ExportResult:
    corrected_path: Path  # same format as the input: annotated .xlsx, or the cleaned .csv
    apply: ApplyResult
    after: dict[str, Any]
    reports: dict[str, Path] = field(default_factory=dict)
    after_run: RunResult | None = None  # validation of the corrected file
    datasets: dict[str, dict[str, str]] = field(default_factory=dict)  # sheet -> {format: file name}
    kept_as_text: list[dict[str, Any]] = field(default_factory=list)
    to_cleaned_row: dict[str, dict[int, int]] = field(default_factory=dict)  # original row -> cleaned row
    original_copy: Path | None = None
    config_path: Path | None = None
    out_dir: Path | None = None
    verification: dict[str, Any] | None = None  # checks run on the exported files
    manifest_path: Path | None = None
    schema_path: Path | None = None
    script_path: Path | None = None


@dataclass
class PreparedResult:
    run: RunResult
    exported: ExportResult

    @property
    def report(self) -> Path:
        return self.exported.reports["json"]

    @property
    def unresolved(self) -> list[dict[str, Any]]:
        return self.exported.after.get("unresolved", [])

    def dataset_path(self, fmt: str = "csv", sheet: str | None = None) -> Path:
        files = self.exported.datasets
        name = files[sheet or next(iter(files))][fmt]
        return (self.exported.out_dir or self.report.parent) / name


def run_checks(
    path: str | Path,
    rules_text: str,
    filename: str | None = None,
    run_id: str | None = None,
) -> RunResult:
    rules = load_rules(rules_text)
    wb = read_workbook(path, filename, rules.source)
    return run_on_workbook(wb, rules, rules_text, run_id)


def run_on_workbook(wb: WorkbookData, rules: RuleSet, rules_text: str, run_id: str | None = None,
                    profiles: dict[str, SheetProfile] | None = None) -> RunResult:
    """`profiles`: reuse a profile computed with the same null tokens (profiling is the slow part)."""
    nulls = normalise_nulls(rules.null_values)
    profiles = profiles or profile_workbook(wb, nulls)
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
    mode: str | None = None,
) -> int:
    """Approve/reject suggestions by id, approve every *safe* one (mode="safe"), or approve all at
    or above min_confidence.

    mode="safe" is Safe Auto-Clean: only fixes classified "safe" (lossless formatting, unambiguous
    conversions, user-declared aliases, and exact-duplicate removal when the policy allows it).
    Bulk approval by confidence never deletes rows unless include_row_deletions is set.
    """
    if mode not in (None, "safe"):
        raise ValueError("mode must be None or 'safe'")
    approve, reject = set(approve), set(reject)
    changed = 0
    for s in run.suggestions:
        target = None
        bulk_ok = include_row_deletions or s.action != "delete_row"
        if s.id in reject:
            target = REJECTED
        elif s.id in approve or (min_confidence is not None and s.confidence >= min_confidence and bulk_ok):
            target = APPROVED
        elif mode == "safe" and s.safety == "safe" and s.status == PENDING:
            target = APPROVED
        if target and s.status != target:
            s.status, s.decided_by, s.decided_at = target, approver, now_iso()
            changed += 1
    if mode == "safe":
        run.auto_cleaned = True
    apply_pre_export(run.issues, run.suggestions)
    return changed


def export(run: RunResult, out_dir: str | Path, annotate: bool = True, write_reports: bool = True,
           formats: tuple[str, ...] = ("csv",), preserve_original: bool = True,
           save_config: bool = True) -> ExportResult:
    """Apply approved fixes, re-validate, and write everything the user and their ML code need.

    Writes (stem = input file name without extension):
      {stem}_cleaned.xlsx                 Excel input: the workbook with changes and unresolved cells marked
      {stem}_cleaned.csv / .parquet       the dataset (one per validated sheet when there are several)
      {stem}_cleaning_report.json/.xlsx/.html
      {stem}_prepare_config.yaml          the exact rules + policy, to clean new batches the same way
      original/{input file}               an untouched copy of the input
    """
    from .dataset import dataset_stem, write_datasets
    from .handoff import build_manifest, recipe_text, reproduce_script, split_hints, verify_outputs
    from .handoff import write_json as handoff_json
    from .report import build_audit
    from .report import write_reports as _write_reports  # local import: jinja2 only needed here

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(run.wb.filename).stem
    existing: dict[str, Path] | None = None
    # The cleaned file keeps the upload's format (a .tsv stays tab-separated); the ML-ready dataset
    # is always a comma CSV, so it can only *be* the cleaned file for a comma-separated .csv.
    text_suffix = Path(run.wb.filename).suffix.lower() if run.wb.file_type == "csv" else ""
    comma_csv = text_suffix == ".csv" and (run.wb.csv_delimiter or ",") == ","
    if cell_writeback(run.wb):
        if run.wb.file_type != "csv":
            name = f"{stem}_cleaned.xlsx"
        elif comma_csv or text_suffix != ".csv":
            name = f"{stem}_cleaned{text_suffix or '.csv'}"
        else:  # e.g. semicolon-separated .csv: keep its delimiter, free the dataset name
            name = f"{stem}_cleaned_as_uploaded.csv"
        corrected = out_dir / name
        applied = apply_corrections(run.wb, run.suggestions, corrected, annotate=annotate)
        # Re-validate the corrected file with the same rules: proves what the fixes achieved.
        # Read it under the original filename: a CSV's sheet name comes from its file name.
        wb2 = read_workbook(corrected, run.wb.filename, run.rules.source)
    else:
        # JSON, Parquet, MATLAB, archives, samples: correct the tables, export them as CSV, and
        # re-validate those exported files.
        names = {t: dataset_stem(stem, t, len(run.wb.sheets)) for t in run.wb.sheets}
        applied, existing = apply_to_tables(run.wb, run.suggestions, out_dir, names, stem)
        corrected = applied.out_path
        wb2 = read_table_csvs(existing, run.wb)
    after_run = run_on_workbook(wb2, run.rules, run.rules_text, run_id=run.run_id + "-after")
    resolution = finalize(run, applied, after_run.validation.issues, wb2.sheets)
    if annotate and run.wb.file_type == "xlsx" and resolution.unresolved:
        mark_unresolved(corrected, resolution.unresolved)  # visible in the file, not only in the report
        applied.sha256 = sha256_file(corrected)
    after = after_run.summary()
    after["unresolved"] = resolution.unresolved
    after["resolution_counts"] = resolution.counts

    _, to_original = row_maps(run, applied)
    to_cleaned = {sheet: {o: c for c, o in m.items()} for sheet, m in to_original.items()}
    datasets, kept = write_datasets(after_run, out_dir, stem, formats,
                                    corrected_csv=corrected if comma_csv and existing is None else None,
                                    existing_csv=existing)
    result = ExportResult(corrected, applied, after, after_run=after_run, datasets=datasets, kept_as_text=kept,
                          to_cleaned_row=to_cleaned, out_dir=out_dir)
    result.verification = verify_outputs(run, result)
    if preserve_original:
        (out_dir / "original").mkdir(exist_ok=True)
        result.original_copy = out_dir / "original" / run.wb.filename
        shutil.copyfile(run.wb.path, result.original_copy)
    if save_config:
        result.config_path = out_dir / f"{stem}_prepare_config.yaml"
        result.config_path.write_text(recipe_text(run), encoding="utf-8")
    if write_reports:
        audit = build_audit(run, result)
        result.reports = _write_reports(run, result, out_dir, audit)
        result.schema_path = handoff_json({"schema_version": "1.0", "source": run.wb.filename,
                                           "tables": audit["columns"], "split_hints": split_hints(run)},
                                          out_dir / f"{stem}_feature_schema.json")
        if result.config_path:
            result.script_path = out_dir / "prepare_dataset.py"
            result.script_path.write_text(reproduce_script(run, result.config_path.name, formats), encoding="utf-8")
        files = {"corrected": corrected, "config": result.config_path, "feature_schema": result.schema_path,
                 "script": result.script_path, **{f"report_{k}": p for k, p in result.reports.items()}}
        result.manifest_path = handoff_json(build_manifest(run, result, audit, files),
                                            out_dir / f"{stem}_manifest.json")
    return result


def prepare(
    path: str | Path,
    out_dir: str | Path = "prepared",
    target: str | None = None,
    config: str | Path | None = None,
    remove_exact_duplicates: bool | None = None,
    target_labels: list[str] | None = None,
    formats: tuple[str, ...] = ("csv",),
    filename: str | None = None,
    approver: str = "safe-auto-clean",
    source: SourceOptions | dict | None = None,
) -> PreparedResult:
    """One click: read (any supported format), profile, validate, apply Safe Auto-Clean,
    re-validate and export.

    Review-required items (ambiguous dates, fuzzy matches, near-duplicates, anomalies, ...) are
    left pending and listed in the report; nothing is imputed, scaled, dropped or deleted unless it
    is safe. `config` re-uses a saved recipe ({stem}_prepare_config.yaml): its source options,
    rules, policy and recorded review decisions. `source` chooses sheets, tables, JSON records,
    a row sample, ... (saved into the recipe).
    """
    src = SourceOptions.model_validate(source) if isinstance(source, dict) else source
    recipe = None
    if config is not None:
        rules_text = Path(config).read_text(encoding="utf-8")
        recipe = load_rules(rules_text)
        src = recipe.source.model_copy(update=src.model_dump(exclude_unset=True)) if src else recipe.source
    src = src or SourceOptions()
    wb = read_workbook(path, filename, src)
    profiles = None
    if config is None:
        profiles = profile_workbook(wb)
        rules_text = suggest_rules(profiles, for_ml=True)
    columns = {name: [str(c) for c in s.df.columns] for name, s in wb.sheets.items()}
    rules_text = with_prepare_options(rules_text, columns, target, remove_exact_duplicates, target_labels, source=src)
    rules = load_rules(rules_text)
    same_nulls = normalise_nulls(rules.null_values) == normalise_nulls(None)
    run = run_on_workbook(wb, rules, rules_text, profiles=profiles if same_nulls else None)
    decide(run, mode="safe", approver=approver)
    if recipe is not None:
        run.recipe_decisions = apply_recipe_decisions(run, recipe)
    return PreparedResult(run, export(run, out_dir, formats=formats))


def apply_recipe_decisions(run: RunResult, recipe: RuleSet) -> dict[str, int]:
    """Re-apply review decisions saved in a recipe; count how many still matched this data."""
    ids = {s.id for s in run.suggestions}
    approve = [i for i in recipe.decisions.approve if i in ids]
    reject = [i for i in recipe.decisions.reject if i in ids]
    decide(run, approve=approve, reject=reject, approver="recipe")
    anomalies = {a.id: a for a in run.anomalies}
    applied_anomalies = 0
    for aid, status in recipe.decisions.anomalies.items():
        if aid in anomalies:
            anomalies[aid].status = status
            applied_anomalies += 1
    apply_pre_export(run.issues, run.suggestions)
    total = len(recipe.decisions.approve) + len(recipe.decisions.reject) + len(recipe.decisions.anomalies)
    applied = len(approve) + len(reject) + applied_anomalies
    return {"recorded": total, "applied": applied, "not_applicable": total - applied}


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
