"""Adapters for ordinary tabular formats: CSV/TSV/TXT, Excel, JSON/JSONL, Parquet, Feather/Arrow, ARFF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..reader import (
    HEADER_SCAN_ROWS,
    ReadError,
    WorkbookData,
    _grid_to_sheet,
    count_lines,
    enforce_row_limit,
    malformed,
    native,
    note_excluded,
    read_workbook,
    sha256_file,
    sheet_from_frame,
)
from ..sources import SourceOptions
from .base import (
    Adapter,
    Inspection,
    TableInfo,
    column_kinds,
    frames_workbook,
    safe_table_name,
    summarise,
)


class DelimitedAdapter(Adapter):
    """CSV/TSV/TXT: read natively (cell-level write-back); inspection reads a sample only."""

    name = "delimited"
    extensions = (".csv", ".tsv", ".txt")

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        return read_workbook(path, filename, options)

    def inspect(self, path: Path, filename: str, options: SourceOptions) -> Inspection:
        sample = read_workbook(path, filename, options.model_copy(update={"max_rows": 200}))
        insp = summarise(sample, sample.source_format, path, options)
        lines = count_lines(path)
        for t in insp.tables:
            t.rows, t.rows_exact = max(0, lines - sample.first_sheet.header_row), False
            t.notes.append(f"delimiter {sample.csv_delimiter!r}, encoding {sample.encoding}; "
                           "row count from physical lines (quoted line breaks can make it differ)")
        return insp


class ExcelAdapter(Adapter):
    """.xlsx/.xlsm natively (cell-level write-back); legacy .xls through xlrd (values only)."""

    name = "excel"
    extensions = (".xlsx", ".xlsm", ".xls")

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        if Path(filename).suffix.lower() != ".xls":
            return read_workbook(path, filename, options)
        try:
            import xlrd  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ReadError("Reading .xls needs xlrd: pip install 'ilovecsv[legacy-excel]'") from e
        try:
            frames = pd.read_excel(path, sheet_name=None, header=None, engine="xlrd")
        except Exception as e:  # xlrd raises its own error types for corrupt files
            raise ReadError(f"Could not read the .xls workbook: {e}") from e
        wb = frames_workbook(filename, path, sha256_file(path), "xls")
        wb.warnings.append("Legacy .xls: cell values only (no formulas or formatting); the cleaned copy is "
                           "written as CSV/Parquet.")
        for name, df in frames.items():
            if options.sheets and name not in options.sheets:
                continue
            grid = [[native(v) for v in row] for row in df.itertuples(index=False)]
            sheet = _grid_to_sheet(safe_table_name(str(name)), grid, [], set())
            if sheet is None:
                wb.warnings.append(f"Sheet '{name}' is empty and was skipped.")
                continue
            sheet = _limit(sheet, wb, options)
            wb.sheets[sheet.name] = sheet
            wb.tables_meta[sheet.name] = {"source": f"sheet {name}", "notes": []}
        if not wb.sheets:
            raise ReadError("The workbook has no sheets with data.")
        return wb


def _limit(sheet, wb: WorkbookData, options: SourceOptions):
    enforce_row_limit(sheet.name, len(sheet.df), options)
    if options.max_rows is not None and len(sheet.df) > options.max_rows:
        total = len(sheet.df)
        sheet.df = sheet.df.iloc[: options.max_rows]
        wb.partial = wb.partial or {"rows": options.max_rows, "totals": {}}
        wb.partial.setdefault("totals", {})[sheet.name] = total
    return sheet


def add_frame(wb: WorkbookData, name: str, df: pd.DataFrame, options: SourceOptions, source: str,
              notes: list[str] | None = None, total_rows: int | None = None,
              text_values: str | None = None) -> None:
    """Add a DataFrame as a table, applying the row limit / sample and recording conversion notes."""
    name = safe_table_name(name)
    total = len(df) if total_rows is None else total_rows
    if options.max_rows is None:
        enforce_row_limit(name, total, options)
    elif total > options.max_rows or len(df) > options.max_rows:
        df = df.head(options.max_rows)
        wb.partial = wb.partial or {"rows": options.max_rows, "totals": {}}
        wb.partial.setdefault("totals", {})[name] = total
    nested = [str(c) for c in df.columns if df[c].map(lambda v: isinstance(v, (list, dict))).any()] if len(df) else []
    sheet = sheet_from_frame(name, df)
    wb.sheets[sheet.name] = sheet
    meta_notes = list(notes or [])
    if nested:
        meta_notes.append(f"columns holding lists/objects are stored as JSON text: {', '.join(nested)} "
                          "(expand them in your pipeline if needed)")
    meta_notes.append("row numbers: header = row 1, so source record k (0-based) is row k + 2")
    wb.tables_meta[sheet.name] = {"source": source, "notes": meta_notes, "rows_in_source": total,
                                  "text_values": text_values}


# ----------------------------------------------------------------------------- JSON / JSONL


def _records_candidates(obj: Any, path: str = "", depth: int = 0) -> list[tuple[str, list]]:
    """Arrays of objects inside a JSON document (where records usually live)."""
    out: list[tuple[str, list]] = []
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj[:50]):
        out.append((path or "$", obj))
    elif isinstance(obj, dict) and depth < 4:
        for k, v in obj.items():
            out += _records_candidates(v, f"{path}.{k}" if path else str(k), depth + 1)
    return out


def _follow(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise ReadError(f"json_records path '{dotted}' not found (stopped at '{part}').")
    return cur


class JSONAdapter(Adapter):
    name = "json"
    extensions = (".json", ".jsonl", ".ndjson")

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        suffix = Path(filename).suffix.lower()
        wb = frames_workbook(filename, path, sha256_file(path), suffix.lstrip("."))
        stem = Path(filename).stem
        if suffix in (".jsonl", ".ndjson"):
            records, bad = [], []
            limit = None if options.max_rows is None else options.max_rows
            total = 0
            last_line = 0
            with open(path, encoding=options.encoding or "utf-8-sig", errors="replace") as f:
                for n, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    total, last_line = total + 1, n
                    if limit is not None and len(records) >= limit:
                        continue  # keep counting for the total
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        bad.append({"line": n, "problem": f"not valid JSON ({e.msg} at column {e.colno})",
                                    "raw": line.rstrip("\r\n")[:300]})
                        continue
                    except RecursionError:
                        bad.append({"line": n, "problem": "nested too deeply to read safely", "raw": line[:300]})
                        continue
                    records.append(rec if isinstance(rec, dict) else {"value": rec})
            if bad and bad[-1]["line"] == last_line and not bad[-1]["raw"].rstrip().endswith("}"):
                bad[-1]["problem"] = "truncated record at the end of the file (cut off before its closing brace)"
            if bad and options.bad_records != "skip":
                raise malformed("JSON lines", bad, valid=len(records), total=total)
            if not records:
                raise ReadError("No JSON records found.")
            df, collisions = flatten_records(records)
            add_frame(wb, stem, df, options, source=f"{filename} (one record per line)", text_values="dates",
                      notes=["nested objects flattened into dotted column names (a.b.c)", *collisions],
                      total_rows=total)
            wb.warnings += collisions
            note_excluded(wb, bad, "malformed JSON line(s)")
            return wb
        try:
            with open(path, encoding=options.encoding or "utf-8-sig") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ReadError(f"Not valid JSON: {e}. Nothing was imported.") from e
        except RecursionError as e:
            raise ReadError("The JSON document is nested too deeply to read safely. Nothing was imported.") from e
        if options.json_records:
            target = _follow(doc, options.json_records)
            if not isinstance(target, list):
                raise ReadError(f"json_records '{options.json_records}' is not an array.")
            candidates = [(options.json_records, target)]
        elif isinstance(doc, list):
            candidates = [("$", doc)]
        else:
            candidates = _records_candidates(doc)
        if not candidates:
            if isinstance(doc, dict):
                candidates = [("$", [doc])]
                wb.warnings.append("No array of records found: the whole object is treated as one record.")
            else:
                raise ReadError("The JSON document holds no records.")
        if len(candidates) > 1:
            wb.warnings.append(f"Several arrays of records found ({', '.join(p for p, _ in candidates)}); each "
                               "became its own table. Set json_records to choose one.")
        for where, items in candidates:
            recs = [x if isinstance(x, dict) else {"value": x} for x in items]
            df, collisions = flatten_records(recs)
            name = stem if len(candidates) == 1 else f"{stem}.{where}"
            add_frame(wb, name, df, options, source=f"{filename} → {where}", text_values="dates",
                      notes=[f"records taken from '{where}'", "nested objects flattened into dotted column names",
                             *collisions])
            wb.warnings += collisions
        return wb


def flatten_records(records: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Flatten nested objects into dotted names (customer.city). If a flattened name collides with a
    key that already contains the dot ({"a.b": 1, "a": {"b": 2}}), pandas would silently keep one of
    the two values; here the nested one is named with '/' (a/b) instead, and the collision is reported."""
    paths: dict[tuple[str, ...], None] = {}

    def walk(obj: dict, prefix: tuple[str, ...], out: dict[tuple[str, ...], Any]) -> None:
        for k, v in obj.items():
            p = (*prefix, str(k))
            if isinstance(v, dict) and v:
                walk(v, p, out)
            else:
                out[p] = v

    flat_rows = []
    for rec in records:
        row: dict[tuple[str, ...], Any] = {}
        walk(rec, (), row)
        flat_rows.append(row)
        paths.update(dict.fromkeys(row))
    by_name: dict[str, list[tuple[str, ...]]] = {}
    for p in paths:
        by_name.setdefault(".".join(p), []).append(p)
    names, notes = {}, []
    for dotted, ps in by_name.items():
        ps = sorted(ps, key=len)
        names[ps[0]] = dotted
        for p in ps[1:]:
            names[p] = "/".join(p)
            notes.append(f"field name collision: '{dotted}' exists as a key and as a nested path "
                         f"{' -> '.join(p)}; both kept, the nested one as '{names[p]}'")
    df = pd.DataFrame([{names[p]: v for p, v in row.items()} for row in flat_rows], columns=list(names.values()))
    return df, notes


# ----------------------------------------------------------------------------- Parquet / Arrow


class ParquetAdapter(Adapter):
    name = "parquet"
    extensions = (".parquet", ".pq")

    def _file(self, path: Path):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise ReadError("Parquet needs pyarrow: pip install 'ilovecsv[parquet]'") from e
        try:
            return pq.ParquetFile(path)
        except Exception as e:
            raise ReadError(f"Not a readable Parquet file: {e}") from e

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        pf = self._file(path)
        total = pf.metadata.num_rows
        wb = frames_workbook(filename, path, sha256_file(path), "parquet")
        if options.max_rows is None:
            enforce_row_limit(Path(filename).stem, total, options)
            df = pf.read().to_pandas()
        else:
            import pyarrow as pa

            batches, got = [], 0
            for b in pf.iter_batches(batch_size=min(65_536, options.max_rows)):
                batches.append(b)
                got += b.num_rows
                if got >= options.max_rows:
                    break
            df = pa.Table.from_batches(batches, schema=pf.schema_arrow).to_pandas() if batches else pd.DataFrame()
        add_frame(wb, Path(filename).stem, df, options, source=f"{filename} ({pf.metadata.num_row_groups} row group(s))",
                  notes=[f"schema: {', '.join(f'{f.name}: {f.type}' for f in pf.schema_arrow)}"[:500]],
                  total_rows=total)
        return wb

    def inspect(self, path: Path, filename: str, options: SourceOptions) -> Inspection:
        pf = self._file(path)
        insp = Inspection(filename=filename, format="parquet", size_bytes=path.stat().st_size,
                          row_limit=options.row_limit)
        head = next(pf.iter_batches(batch_size=5), None)
        sample = [] if head is None else [{k: native(v) for k, v in r.items()} for r in head.to_pylist()]
        insp.tables.append(TableInfo(
            name=Path(filename).stem, rows=pf.metadata.num_rows, rows_exact=True,
            columns=[{"name": f.name, "kind": str(f.type)} for f in pf.schema_arrow], sample=sample,
            source=filename, notes=["metadata read without loading the data"]))
        return insp


class ArrowAdapter(Adapter):
    name = "arrow"
    extensions = (".feather", ".arrow", ".ipc")

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        try:
            import pyarrow.feather as feather
        except ImportError as e:  # pragma: no cover
            raise ReadError("Feather/Arrow needs pyarrow: pip install 'ilovecsv[parquet]'") from e
        try:
            table = feather.read_table(path)
        except Exception as e:
            raise ReadError(f"Not a readable Feather/Arrow file: {e}") from e
        wb = frames_workbook(filename, path, sha256_file(path), Path(filename).suffix.lstrip("."))
        add_frame(wb, Path(filename).stem, table.to_pandas(), options, source=filename, total_rows=table.num_rows)
        return wb


class ARFFAdapter(Adapter):
    """Weka ARFF (classic UCI datasets). Nominal values are decoded; '?' becomes missing."""

    name = "arff"
    extensions = (".arff",)

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        from scipy.io import arff

        try:
            data, meta = arff.loadarff(str(path))
        except NotImplementedError as e:
            raise ReadError(f"This ARFF file uses a feature the reader does not support: {e}") from e
        except Exception as e:
            raise ReadError(f"Not a readable ARFF file: {e}") from e
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                df[col] = df[col].replace({"?": None})
        wb = frames_workbook(filename, path, sha256_file(path), "arff")
        types = ", ".join(f"{n}: {meta[n][0]}" for n in meta.names())
        add_frame(wb, meta.name or Path(filename).stem, df, options, source=filename,
                  notes=[f"ARFF relation '{meta.name}'", f"attribute types: {types}"[:500]])
        return wb


__all__ = ["DelimitedAdapter", "ExcelAdapter", "JSONAdapter", "ParquetAdapter", "ArrowAdapter", "ARFFAdapter",
           "add_frame", "column_kinds", "HEADER_SCAN_ROWS"]
