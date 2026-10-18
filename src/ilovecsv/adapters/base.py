"""The adapter contract: every format becomes clearly defined tables, plus metadata about how.

An adapter does three things:
  inspect(path)  structure first, without processing everything: tables, row counts (exact or
                 estimated), columns, a few sample rows, documentation, parts it cannot tabulate;
  load(path)     the tables as WorkbookData (raw values, rows numbered like a sheet: header row 1);
  preview        the sample rows inside an inspection.
Export is shared (dataset.py): every table leaves as CSV/Parquet, whatever it came from.

Adapters never guess meaning. When a structure has no safe table form (a 3-D array, a struct of
variable-length series with no known layout), it is listed under `unsupported` with the reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models import jsonable
from ..reader import WorkbookData
from ..sources import SourceOptions

BYTES_PER_CELL = 72  # rough in-memory cost of one raw cell in the engine (object dtype + value)
SAMPLE_ROWS = 5


@dataclass
class TableInfo:
    name: str
    rows: int | None
    rows_exact: bool
    columns: list[dict[str, Any]]  # {name, kind}
    sample: list[dict[str, Any]]
    source: str = ""  # sheet, archive member, dataset path, ...
    notes: list[str] = field(default_factory=list)


@dataclass
class Inspection:
    filename: str
    format: str
    size_bytes: int
    tables: list[TableInfo] = field(default_factory=list)
    tree: list[dict[str, Any]] = field(default_factory=list)  # hierarchical formats: what is inside
    docs: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_limit: int = 0
    excluded: list[dict[str, Any]] = field(default_factory=list)  # malformed records / damaged parts found

    @property
    def memory_estimate_mb(self) -> float:
        cells = sum((t.rows or 0) * max(1, len(t.columns)) for t in self.tables)
        return round(cells * BYTES_PER_CELL / 1e6, 1)

    @property
    def full_processing_ok(self) -> bool:
        return all((t.rows or 0) <= self.row_limit for t in self.tables)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["memory_estimate_mb"] = self.memory_estimate_mb
        d["full_processing_ok"] = self.full_processing_ok
        return d


class Adapter:
    name = "base"
    extensions: tuple[str, ...] = ()

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:  # pragma: no cover
        raise NotImplementedError

    def inspect(self, path: Path, filename: str, options: SourceOptions) -> Inspection:
        """Default: load a sample and summarise it (cheap formats override with metadata reads)."""
        sample_opts = options.model_copy(update={"max_rows": min(options.max_rows or 1000, 1000)})
        wb = self.load(path, filename, sample_opts)
        insp = summarise(wb, self.name, path, options)
        partial = wb.partial or {}
        for t in insp.tables:
            total = (partial.get("totals") or {}).get(t.name)
            if total is not None:
                t.rows, t.rows_exact = total, True
        return insp


def column_kinds(wb_sheet) -> list[dict[str, Any]]:
    out = []
    for col in wb_sheet.df.columns:
        kinds = {type(v).__name__ for v in wb_sheet.df[col].head(200) if v is not None}
        out.append({"name": str(col), "kind": ", ".join(sorted(kinds)) or "empty"})
    return out


def summarise(wb: WorkbookData, fmt: str, path: Path, options: SourceOptions) -> Inspection:
    insp = Inspection(filename=wb.filename, format=wb.source_format or fmt, size_bytes=Path(path).stat().st_size,
                      docs=wb.docs, unsupported=wb.unsupported, warnings=list(wb.warnings),
                      row_limit=options.row_limit, excluded=list(wb.excluded_records))
    for name, sheet in wb.sheets.items():
        meta = wb.tables_meta.get(name, {})
        sample = [{str(k): jsonable(v) for k, v in row.items()}
                  for row in sheet.df.head(SAMPLE_ROWS).to_dict("records")]
        totals = (wb.partial or {}).get("totals")
        exact = not wb.partial or (totals is not None and name not in totals)  # only sampled tables are estimates
        insp.tables.append(TableInfo(name=name, rows=len(sheet.df), rows_exact=exact,
                                     columns=column_kinds(sheet), sample=sample,
                                     source=str(meta.get("source", "")), notes=list(meta.get("notes", []))))
    insp.tree = list(wb.tree)
    return insp


def frames_workbook(filename: str, path: Path, digest: str, source_format: str) -> WorkbookData:
    return WorkbookData(filename=filename, path=path, file_type="frames", sha256=digest, sheets={},
                        source_format=source_format)


def safe_table_name(name: str) -> str:
    """Table names end up in cell refs (table!C5) and file names, so no '!', path separators or ':'."""
    for ch in "!/\\:":
        name = name.replace(ch, "_")
    return name.strip() or "table"
