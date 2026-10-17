"""Load .xlsx / .csv files into DataFrames that remember where every cell came from.

Each sheet becomes an object-dtype DataFrame whose index is the Excel row
number and whose columns map back to Excel column letters, so every issue can
point at a real cell such as ``Orders!D14``. Values are kept raw (no type
coercion), so IDs like ``00123`` survive and the profiler sees what is really in
the file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

SUPPORTED = {".xlsx", ".csv"}
HEADER_SCAN_ROWS = 30
MAX_ROWS = 250_000


class ReadError(Exception):
    """The file cannot be read as a workbook."""


@dataclass
class SheetData:
    name: str
    df: pd.DataFrame  # index = Excel row number, dtype=object
    header_row: int
    col_letters: dict[str, str]
    formula_cells: set[str] = field(default_factory=set)  # coordinates such as "H5"
    dropped_blank_rows: list[int] = field(default_factory=list)
    dropped_blank_cols: list[str] = field(default_factory=list)

    def coord(self, xl_row: int, column: str) -> str:
        return f"{self.col_letters[column]}{xl_row}"

    def cell_ref(self, xl_row: int, column: str) -> str:
        return f"{self.name}!{self.coord(xl_row, column)}"

    def row_ref(self, xl_row: int) -> str:
        first = next(iter(self.col_letters.values()), "A")
        return f"{self.name}!{first}{xl_row}"

    def is_formula(self, xl_row: int, column: str) -> bool:
        return self.coord(xl_row, column) in self.formula_cells


@dataclass
class WorkbookData:
    filename: str
    path: Path
    file_type: str  # "xlsx" | "csv"
    sha256: str
    sheets: dict[str, SheetData]
    warnings: list[str] = field(default_factory=list)
    csv_delimiter: str | None = None

    @property
    def first_sheet(self) -> SheetData:
        return next(iter(self.sheets.values()))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_workbook(path: str | Path, filename: str | None = None) -> WorkbookData:
    path = Path(path)
    filename = filename or path.name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ReadError(f"Unsupported file type '{suffix}'. Use .xlsx or .csv.")
    digest = sha256_file(path)
    if suffix == ".csv":
        sheet, delimiter = _read_csv(path, Path(filename).stem)
        return WorkbookData(filename, path, "csv", digest, {sheet.name: sheet}, csv_delimiter=delimiter)
    return _read_xlsx(path, filename, digest)


# --------------------------------------------------------------------------- xlsx


def _read_xlsx(path: Path, filename: str, digest: str) -> WorkbookData:
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile as e:
        raise ReadError("The file is not a valid .xlsx workbook.") from e
    if any(n.startswith(("xl/charts/", "xl/drawings/")) for n in names):
        warnings.append(
            "The workbook contains charts or drawings; openpyxl may not preserve them "
            "in the corrected copy."
        )
    try:
        wb_values = load_workbook(path, data_only=True)
        wb_formulas = load_workbook(path, data_only=False)
    except Exception as e:  # openpyxl raises many types for corrupt files
        raise ReadError(f"Could not open workbook: {e}") from e

    sheets: dict[str, SheetData] = {}
    for ws in wb_values.worksheets:
        ws_f = wb_formulas[ws.title]
        formula_cells = {
            c.coordinate for row in ws_f.iter_rows() for c in row if c.data_type == "f"
        }
        grid = [list(r) for r in ws.iter_rows(values_only=True)]
        merged = [
            (m.min_row, m.max_row, m.min_col, m.max_col) for m in ws.merged_cells.ranges
        ]
        sheet = _grid_to_sheet(ws.title, grid, merged, formula_cells)
        if sheet is None:
            warnings.append(f"Sheet '{ws.title}' is empty and was skipped.")
            continue
        sheets[sheet.name] = sheet
    if not sheets:
        raise ReadError("The workbook has no sheets with data.")
    return WorkbookData(filename, path, "xlsx", digest, sheets, warnings)


# --------------------------------------------------------------------------- csv


def _read_csv(path: Path, stem: str) -> tuple[SheetData, str]:
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ReadError("The CSV file contains binary data.")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 decodes almost anything
        raise ReadError("Could not decode the CSV file.")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    grid: list[list[Any]] = [
        [None if c == "" else c for c in row]
        for row in csv.reader(io.StringIO(text), delimiter=delimiter)
    ]
    name = (stem or "data")[:31]
    sheet = _grid_to_sheet(name, grid, [], set())
    if sheet is None:
        raise ReadError("The CSV file has no data.")
    return sheet, delimiter


# --------------------------------------------------------------------------- shared


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _detect_header(grid: list[list[Any]]) -> int | None:
    scan = grid[:HEADER_SCAN_ROWS]
    widths = [sum(not _blank(v) for v in r) for r in scan]
    if not widths or max(widths) == 0:
        return None
    max_width = max(widths)
    for i, row in enumerate(scan):
        filled = [v for v in row if not _blank(v)]
        if len(filled) < max(1, 0.5 * max_width):
            continue
        text = sum(isinstance(v, str) for v in filled)
        if text / len(filled) >= 0.5:
            return i
    return next(i for i, w in enumerate(widths) if w)


def _grid_to_sheet(
    name: str,
    grid: list[list[Any]],
    merged: list[tuple[int, int, int, int]],
    formula_cells: set[str],
) -> SheetData | None:
    if len(grid) > MAX_ROWS:
        raise ReadError(f"Sheet '{name}' has more than {MAX_ROWS} rows.")
    width = max((len(r) for r in grid), default=0)
    grid = [r + [None] * (width - len(r)) for r in grid]
    h = _detect_header(grid)
    if h is None:
        return None
    header_xl_row = h + 1

    # Merged header cells keep their value only in the top-left cell: spread it.
    header = list(grid[h])
    for min_r, max_r, min_c, max_c in merged:
        if min_r <= header_xl_row <= max_r:
            top_left = grid[min_r - 1][min_c - 1]
            for c in range(min_c - 1, max_c):
                if _blank(header[c]):
                    header[c] = top_left

    body = grid[h + 1 :]
    # Drop trailing blank rows (formatting often extends the used range).
    while body and all(_blank(v) for v in body[-1]):
        body.pop()

    keep_cols: list[int] = []
    dropped_cols: list[str] = []
    for c in range(width):
        if _blank(header[c]) and all(_blank(r[c]) for r in body):
            if c < max((i for i, v in enumerate(header) if not _blank(v)), default=-1):
                dropped_cols.append(get_column_letter(c + 1))
            continue
        keep_cols.append(c)

    names: list[str] = []
    seen: dict[str, int] = {}
    for c in keep_cols:
        v = header[c]
        base = str(v).strip() if not _blank(v) else f"column_{get_column_letter(c + 1)}"
        if base in seen:
            seen[base] += 1
            base = f"{base}.{seen[base]}"
        else:
            seen[base] = 0
        names.append(base)

    rows: list[list[Any]] = []
    index: list[int] = []
    dropped_rows: list[int] = []
    for offset, r in enumerate(body):
        xl_row = header_xl_row + 1 + offset
        values = [r[c] for c in keep_cols]
        if all(_blank(v) for v in values):
            dropped_rows.append(xl_row)
            continue
        rows.append(values)
        index.append(xl_row)

    df = pd.DataFrame(rows, columns=names, index=pd.Index(index, name="xl_row"), dtype=object)
    if df.empty and not names:
        return None
    letters = {n: get_column_letter(c + 1) for n, c in zip(names, keep_cols, strict=True)}
    return SheetData(
        name=name,
        df=df,
        header_row=header_xl_row,
        col_letters=letters,
        formula_cells=formula_cells,
        dropped_blank_rows=dropped_rows,
        dropped_blank_cols=dropped_cols,
    )
