"""Write the cleaned dataset files an ML pipeline reads: CSV always, typed Parquet on request.

CSV conventions: one header row, ISO-8601 dates, plain numbers (no thousands separators),
lowercase true/false, and an empty field for every missing value, so
``pandas.read_csv(path)`` needs no extra options.
"""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ml import sheet_schema, typed_frame
from .values import normalise_nulls

if TYPE_CHECKING:
    from .pipeline import RunResult

FORMATS = ("csv", "parquet")


def dataset_stem(stem: str, sheet: str, n_sheets: int) -> str:
    if n_sheets == 1:
        return f"{stem}_cleaned"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sheet).strip("_") or "sheet"
    return f"{stem}_cleaned" if safe == stem else f"{stem}_{safe}_cleaned"


def csv_cell(v: Any) -> str:
    """Only real missing values become empty fields: placeholder text such as 'N/A' is written as it
    is unless its fix was approved, so the file shows exactly what was (and was not) cleaned."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, datetime):
        return v.date().isoformat() if v.time() == datetime.min.time() else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    return str(v)


def write_table_csv(df, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([str(c) for c in df.columns])
        for row in df.itertuples(index=False):
            w.writerow([csv_cell(v) for v in row])


def write_datasets(after: RunResult, out_dir: Path, stem: str, formats: tuple[str, ...],
                   corrected_csv: Path | None = None, existing_csv: dict[str, Path] | None = None
                   ) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """Write one dataset per validated sheet. Returns ({sheet: {format: file name}}, kept_as_text).

    corrected_csv: CSV input, where the corrected file already is the dataset.
    existing_csv: tables already written as cleaned CSVs (non-spreadsheet sources)."""
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"unknown dataset format(s): {sorted(unknown)}; use {list(FORMATS)}")
    nulls = normalise_nulls(after.rules.null_values)
    sheets = [s.sheet or after.wb.first_sheet.name for s in after.rules.sheets]
    sheets = [s for s in sheets if s in after.wb.sheets]
    by_name = {(s.sheet or after.wb.first_sheet.name): s for s in after.rules.sheets}
    files: dict[str, dict[str, str]] = {}
    kept_as_text: list[dict[str, Any]] = []
    for name in sheets:
        sheet = after.wb.sheets[name]
        base = dataset_stem(stem, name, len(sheets))
        entry: dict[str, str] = {}
        if "csv" in formats or not formats:
            if existing_csv and name in existing_csv:
                entry["csv"] = existing_csv[name].name
            elif corrected_csv is not None:
                entry["csv"] = corrected_csv.name  # CSV input: the corrected file is the dataset
            else:
                path = out_dir / f"{base}.csv"
                write_table_csv(sheet.df, path)
                entry["csv"] = path.name
        schema = sheet_schema(after.profiles[name], by_name.get(name))
        frame, kept = typed_frame(sheet, schema, nulls)
        kept_as_text.extend(kept)
        if "parquet" in formats:
            try:
                import pyarrow  # noqa: F401
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise RuntimeError("Parquet export needs pyarrow: pip install 'ilovecsv[parquet]'") from e
            path = out_dir / f"{base}.parquet"
            frame.to_parquet(path, index=False)
            entry["parquet"] = path.name
        files[name] = entry
    return files, kept_as_text
