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
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

NATIVE = {".xlsx", ".xlsm", ".csv", ".tsv", ".txt"}  # read here; other formats go through adapters
HEADER_SCAN_ROWS = 30


class ReadError(Exception):
    """The file cannot be read as a dataset."""


class LargeDatasetError(ReadError):
    """The table is larger than the full-processing limit; inspect it, or process a sample."""

    status = "too-large"


class UnsupportedFormatError(ReadError):
    """The file type is not one IloveCSV reads."""

    status = "unsupported"


class ContentMismatchError(ReadError):
    """The content does not match the extension (e.g. JSON saved as .csv)."""

    status = "content-mismatch"


class DamagedFileError(ReadError):
    """The container itself is broken (truncated, failed CRC, unreadable structure). `problems` lists
    damaged parts when the rest is intact (e.g. one archive member), so a partial import is possible."""

    status = "damaged"

    def __init__(self, message: str, problems: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.problems = problems or []


class MalformedRecordsError(ReadError):
    """Some records cannot be parsed. Nothing is excluded without consent: re-run with
    bad_records='skip' to import the valid records; the excluded ones are then listed in the report."""

    status = "malformed-records"

    def __init__(self, message: str, problems: list[dict[str, Any]]):
        super().__init__(message)
        self.problems = problems


ReadError.status = "unreadable"  # type: ignore[attr-defined]


def partial_text(partial: dict[str, Any] | None) -> str:
    """Why an import is not complete, or '' when it is."""
    if not partial:
        return ""
    parts = []
    if partial.get("rows"):
        parts.append(f"only the first {partial['rows']} rows of each table were processed (max_rows)")
    if partial.get("excluded_records"):
        parts.append(f"{partial['excluded_records']} malformed record(s) / damaged part(s) were excluded on request "
                     "(listed in the report)")
    return "; ".join(parts) or "only part of the source was imported"


def where(p: dict[str, Any]) -> str:
    """'line 3', 'data.csv', or 'data.csv line 3' for an excluded record or member."""
    parts = [str(p["member"])] if p.get("member") else []
    if p.get("line") is not None:
        parts.append(f"line {p['line']}")
    return " ".join(parts) or "?"


def malformed(what: str, problems: list[dict[str, Any]], valid: int, total: int) -> MalformedRecordsError:
    shown = "; ".join(f"{where(p)}: {p['problem']}" for p in problems[:8])
    more = f" (and {len(problems) - 8} more)" if len(problems) > 8 else ""
    return MalformedRecordsError(
        f"{len(problems)} of {total} {what} cannot be parsed: {shown}{more}. Nothing was imported. To import "
        f"the {valid} valid record(s) and exclude these (each is listed in the report, and the output is marked "
        "PARTIAL), re-run with bad_records: skip (CLI --skip-bad-records).", problems)


def note_excluded(wb: WorkbookData, problems: list[dict[str, Any]], what: str) -> None:
    """Record records excluded with consent: listed in the report, and the output marked partial."""
    if not problems:
        return
    wb.excluded_records += problems
    wb.partial = {**(wb.partial or {"totals": {}}), "excluded_records": len(wb.excluded_records)}
    wb.warnings.append(f"PARTIAL: {len(problems)} {what} excluded on request (bad_records: skip): "
                       + "; ".join(where(p) for p in problems[:20]) + ". They are listed in the report.")


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
    """A source file as one or more named tables ("sheets"), whatever its original format."""

    filename: str
    path: Path
    file_type: str  # "xlsx" | "csv" (cell-level write-back) | "frames" (any other adapter)
    sha256: str
    sheets: dict[str, SheetData]
    warnings: list[str] = field(default_factory=list)
    csv_delimiter: str | None = None
    source_format: str = ""  # e.g. "xlsx", "csv", "jsonl", "parquet", "zip", "mat", "nasa_battery_mat"
    tables_meta: dict[str, dict[str, Any]] = field(default_factory=dict)  # per table: source, conversion notes
    docs: list[dict[str, Any]] = field(default_factory=list)  # documentation found next to the data
    unsupported: list[dict[str, Any]] = field(default_factory=list)  # readable parts with no safe table form
    partial: dict[str, Any] | None = None  # set when only a sample of rows was processed
    encoding: str | None = None  # text sources
    tree: list[dict[str, Any]] = field(default_factory=list)  # hierarchical sources: what is inside
    excluded_records: list[dict[str, Any]] = field(default_factory=list)  # malformed records skipped on request

    @property
    def first_sheet(self) -> SheetData:
        return next(iter(self.sheets.values()))


def native(v: Any) -> Any:
    """Plain Python value for the engine: numpy scalars, Timestamps, NaN/NaT, bytes, nested data."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, float):
        return None if v != v else v
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime().replace(tzinfo=None) if v.tzinfo else v.to_pydatetime()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    if hasattr(v, "dtype") and hasattr(v, "item") and getattr(v, "ndim", 0) == 0:  # numpy scalar
        if str(v.dtype).startswith("datetime64"):
            ts = pd.Timestamp(v)
            return None if ts is pd.NaT else ts.to_pydatetime()
        return native(v.item())
    if isinstance(v, (list, tuple, dict)) or (hasattr(v, "tolist") and getattr(v, "ndim", 0) > 0):
        import json

        data = v.tolist() if hasattr(v, "tolist") else v
        return json.dumps(data, ensure_ascii=False, default=str)
    return v


def sheet_from_frame(name: str, df: pd.DataFrame) -> SheetData:
    """Build a table from a DataFrame, numbered like a sheet (header row 1, data from row 2)."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for c in df.columns:
        base = str(c).strip() or "column"
        if base in seen:
            seen[base] += 1
            base = f"{base}.{seen[base]}"
        else:
            seen[base] = 0
        names.append(base)
    data: dict[str, list[Any]] = {}
    for new, col in zip(names, df.columns, strict=True):
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            data[new] = [None if pd.isna(x) else native(pd.Timestamp(x)) for x in s]
        else:
            data[new] = [native(x) for x in s.astype(object).tolist()]
    out = pd.DataFrame(data, columns=names, dtype=object)
    out.index = pd.Index(range(2, len(out) + 2), name="xl_row")
    letters = {n: get_column_letter(i + 1) for i, n in enumerate(names)}
    return SheetData(name=name[:100], df=out, header_row=1, col_letters=letters)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_workbook(path: str | Path, filename: str | None = None, options: Any = None) -> WorkbookData:
    """Read any supported file into tables. `options` is a sources.SourceOptions (or None)."""
    from .sources import SourceOptions

    options = options or SourceOptions()
    path = Path(path)
    filename = filename or path.name
    suffix = Path(filename).suffix.lower()
    if suffix not in NATIVE:
        from .adapters import load_dataset  # JSON, Parquet, ZIP, MATLAB, NumPy, HDF5, ...

        return load_dataset(path, filename, options)
    digest = sha256_file(path)
    if suffix in (".csv", ".tsv", ".txt"):
        _check_text_content(path, suffix)
        if count_lines(path) == 0:
            raise ReadError("The file is empty. Nothing was imported.")
        sheet, delimiter, partial, encoding, problems = _read_csv(path, Path(filename).stem, options)
        if problems and options.bad_records != "skip":
            raise malformed("records", problems, valid=len(sheet.df), total=len(sheet.df) + len(problems))
        wb = WorkbookData(filename, path, "csv", digest, {sheet.name: sheet}, csv_delimiter=delimiter,
                          source_format=suffix.lstrip("."), partial=partial, encoding=encoding)
        if partial:
            wb.warnings.append(f"Only the first {partial['rows']} rows were processed (max_rows).")
        note_excluded(wb, problems, "malformed record(s)")
        return wb
    wb = _read_xlsx(path, filename, digest)
    wb.source_format = suffix.lstrip(".")
    if suffix == ".xlsm":
        wb.warnings.append("Macro-enabled workbook: macros were not run, and the cleaned copy is saved as .xlsx "
                           "without them.")
    if options.sheets:
        missing = [s for s in options.sheets if s not in wb.sheets]
        if missing:
            raise ReadError(f"Sheet(s) not found: {missing}. Available: {list(wb.sheets)}")
        wb.sheets = {k: v for k, v in wb.sheets.items() if k in options.sheets}
    for name, sheet in wb.sheets.items():
        enforce_row_limit(name, len(sheet.df), options)
    return wb


def read_table_csvs(files: dict[str, Path], original: WorkbookData) -> WorkbookData:
    """Re-read cleaned per-table CSVs (non-spreadsheet sources) under their table names, so the
    exported files themselves are what gets re-validated."""
    from .sources import SourceOptions

    opts = SourceOptions(delimiter=",", encoding="utf-8", row_limit=10**12)
    sheets: dict[str, SheetData] = {}
    for table, path in files.items():
        sheet, *_ = _read_csv(Path(path), table, opts)
        sheet.name = table
        sheets[table] = sheet
    return WorkbookData(original.filename, Path(next(iter(files.values()))), "csv",
                        sha256_file(next(iter(files.values()))), sheets, csv_delimiter=",",
                        source_format="csv", encoding="utf-8")


def text_values(wb: WorkbookData, table: str) -> str | None:
    """Which values a table's source can only hold as text, so text there is not a problem:
    "all" (CSV/TSV, also inside archives), "dates" (JSON has no date type), or None (typed formats)."""
    if wb.file_type == "csv":
        return "all"
    return (wb.tables_meta.get(table) or {}).get("text_values")


def enforce_row_limit(table: str, rows: int, options: Any) -> None:
    if options.max_rows is None and rows > options.row_limit:
        raise LargeDatasetError(
            f"Table '{table}' has {rows:,} rows, above the full-processing limit of {options.row_limit:,}. "
            "Inspect it first (ilovecsv inspect), or process a sample with max_rows.")


# --------------------------------------------------------------------------- xlsx


def _read_xlsx(path: Path, filename: str, digest: str) -> WorkbookData:
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            broken = z.testzip()  # reads every part and checks its CRC
    except (zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
        raise DamagedFileError(f"The workbook is damaged or truncated: its zip structure cannot be read ({e}). "
                               "Nothing was imported; re-export it from Excel.") from e
    if broken:
        raise DamagedFileError(f"The workbook is damaged: part '{broken}' fails its CRC check. Nothing was "
                               "imported; re-export it from Excel.")
    if any(n.startswith(("xl/charts/", "xl/drawings/")) for n in names):
        warnings.append(
            "The workbook contains charts or drawings; openpyxl may not preserve them "
            "in the corrected copy."
        )
    try:
        wb_values = load_workbook(path, data_only=True)
        wb_formulas = load_workbook(path, data_only=False)
    except Exception as e:  # openpyxl raises many types for corrupt files
        raise DamagedFileError(f"The workbook is damaged and cannot be opened: {e}. Nothing was imported.") from e

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


def detect_text_encoding(head: bytes, preferred: str | None = None) -> str:
    candidates = [preferred] if preferred else ["utf-8-sig", "cp1252"]
    for enc in candidates:
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
        except LookupError as e:
            raise ReadError(f"Unknown encoding '{enc}'.") from e
    if preferred:
        raise ReadError(f"The file is not valid {preferred} text.")
    return "cp1252"  # pragma: no cover - cp1252 decodes almost anything


def count_lines(path: Path) -> int:
    """Physical line count without decoding (fast, for size estimates)."""
    n, last = 0, b""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (1 if last and last != b"\n" else 0)


def _read_csv(path: Path, stem: str, options: Any = None) -> tuple[SheetData, str, dict[str, Any] | None, str]:
    from .sources import SourceOptions

    options = options or SourceOptions()
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    if b"\x00" in head[:4096]:
        raise ReadError("The CSV file contains binary data.")
    encoding = detect_text_encoding(head, options.encoding)
    sample = head.decode(encoding, errors="replace")
    delimiter = options.delimiter
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    lines = count_lines(path)
    if options.max_rows is None:
        enforce_row_limit(stem, max(0, lines - 1), options)
    limit = None if options.max_rows is None else options.max_rows + HEADER_SCAN_ROWS
    errors = "strict" if options.encoding else "replace"
    records, problems = _parse_csv(path, encoding, errors, delimiter, limit)
    bad_lines = {p["line"] for p in problems}
    # An empty line ([] from the csv module) is layout; a line of empty fields ('""', ',,') is a record.
    grid = [[None if c == "" else c for c in row] for line, row in records if line not in bad_lines and row]
    sheet = _grid_to_sheet(stem or "data", grid, [], set(), keep_blank_rows=True)
    if sheet is None:
        raise ReadError("The CSV file has no data.")
    partial = None
    if options.max_rows is not None and (len(sheet.df) > options.max_rows or limit and len(records) >= limit):
        sheet.df = sheet.df.iloc[: options.max_rows]
        partial = {"rows": len(sheet.df), "estimated_total_rows": max(0, lines - sheet.header_row)}
    return sheet, delimiter, partial, encoding, problems


def _check_text_content(path: Path, suffix: str) -> None:
    """A .csv/.tsv/.txt that actually holds JSON would be misparsed into junk columns: refuse it."""
    import json

    with open(path, "rb") as f:
        head = f.read(1 << 16)
    text = head.decode("utf-8-sig", errors="replace").lstrip()
    if not text.startswith(("{", "[")):
        return
    first = text.splitlines()[0] if text else ""
    try:
        json.loads(text if len(head) < (1 << 16) else first)
    except ValueError:
        if not text.startswith(("[{", '{"')):
            return
    raise ContentMismatchError(f"The content of this {suffix} file is JSON, not delimited text, so reading it as "
                               f"{suffix} would produce junk columns. Rename it to .json (one document) or .jsonl "
                               "(one record per line). Nothing was imported.")


def _parse_csv(path: Path, encoding: str, errors: str, delimiter: str,
               limit: int | None) -> tuple[list[tuple[int, list[str]]], list[dict[str, Any]]]:
    """Records with the physical line they start on, plus the records that cannot be parsed.

    Strict parsing: a quote that is opened and never closed would otherwise swallow every following
    line into one field (silent row loss). That record is reported, and parsing resumes on the next
    line. Records whose field count differs from the file's usual count are reported too."""

    def read(lines, offset: int) -> tuple[int | None, str]:
        reader = csv.reader(lines, delimiter=delimiter, strict=True)
        prev = 0
        try:
            for row in reader:
                records.append((offset + prev + 1, row))
                prev = reader.line_num
                if limit is not None and len(records) >= limit:
                    return None, ""
        except csv.Error as e:
            return offset + prev + 1, str(e)
        return None, ""

    records: list[tuple[int, list[str]]] = []
    problems: list[dict[str, Any]] = []
    with open(path, encoding=encoding, newline="", errors=errors) as f:
        broken, err = read(f, 0)
    if broken is not None:
        with open(path, encoding=encoding, newline="", errors=errors) as f:
            all_lines = f.readlines()
        while broken is not None and len(problems) < 1000:
            what = ("a quote opened on this line is never closed, so it would swallow the following lines into "
                    "one field" if "end of data" in err else f"invalid quoting ({err})")
            problems.append({"line": broken, "problem": what,
                             "raw": all_lines[broken - 1].rstrip("\r\n")[:300] if broken <= len(all_lines) else ""})
            broken, err = read(iter(all_lines[broken:]), broken)
    filled = [r for _, r in records if any(c.strip() for c in r)]
    if filled:
        expected = Counter(len(r) for r in filled[:5000]).most_common(1)[0][0]
        for line, r in records:
            if any(c.strip() for c in r) and len(r) != expected:
                extra = f" (extra: {', '.join(repr(c) for c in r[expected:][:5])})" if len(r) > expected else ""
                problems.append({"line": line, "problem": f"{len(r)} fields, expected {expected}{extra}",
                                 "raw": delimiter.join(r)[:300]})
    problems.sort(key=lambda p: p["line"])
    return records, problems


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
    keep_blank_rows: bool = False,
) -> SheetData | None:
    """keep_blank_rows: every row is a record (delimited text: a line of empty fields is a record with
    missing values, not layout). Spreadsheets drop blank rows, which the profile lists."""
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
    while body and not keep_blank_rows and all(_blank(v) for v in body[-1]):
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
        if all(_blank(v) for v in values) and not keep_blank_rows:
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
