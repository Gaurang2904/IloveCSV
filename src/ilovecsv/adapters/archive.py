"""ZIP archives: datasets often arrive as several files plus a README.

Safety first: the archive is listed before anything is extracted, with limits on the number of
members, the total uncompressed size and the compression ratio (zip bombs), and member paths
that try to escape the extraction folder are refused. Only members in supported formats are
extracted, one at a time, into a temporary folder that is removed afterwards.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from ..reader import ReadError, WorkbookData, sha256_file
from ..sources import SourceOptions
from .base import Adapter, Inspection, frames_workbook, safe_table_name, summarise
from .docs import is_doc_file, read_doc

MAX_MEMBERS = 5_000
MAX_TOTAL_BYTES = 4 * 1024 ** 3
MAX_RATIO = 200


def _member_ok(info: zipfile.ZipInfo) -> str | None:
    """Reason to skip a member, or None."""
    p = PurePosixPath(info.filename)
    if info.is_dir():
        return "directory"
    if p.is_absolute() or ".." in p.parts or ":" in p.parts[0]:
        return "unsafe path (tries to leave the archive folder)"
    if p.name.startswith(".") or "__MACOSX" in p.parts:
        return "system file"
    if info.compress_size and info.file_size > 1 << 20 and info.file_size / info.compress_size > MAX_RATIO:
        return f"suspicious compression ratio ({info.file_size // max(info.compress_size, 1)}:1)"
    return None


def list_members(path: Path) -> tuple[list[zipfile.ZipInfo], list[dict[str, Any]]]:
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except zipfile.BadZipFile as e:
        raise ReadError("Not a valid ZIP archive.") from e
    if len(infos) > MAX_MEMBERS:
        raise ReadError(f"The archive has {len(infos):,} members (limit {MAX_MEMBERS:,}).")
    total = sum(i.file_size for i in infos)
    if total > MAX_TOTAL_BYTES:
        raise ReadError(f"The archive expands to {total / 1e9:.1f} GB (limit {MAX_TOTAL_BYTES / 1e9:.0f} GB).")
    keep, skipped = [], []
    for info in infos:
        reason = _member_ok(info)
        if reason is None:
            keep.append(info)
        elif reason not in ("directory", "system file"):
            skipped.append({"path": info.filename, "reason": reason})
    return keep, skipped


class ZipAdapter(Adapter):
    name = "zip"
    extensions = (".zip",)

    def _classify(self, infos: list[zipfile.ZipInfo]):
        from . import adapter_for  # registry (late import: it imports this module)

        data, docs, other = [], [], []
        for info in infos:
            name = info.filename
            if is_doc_file(name):
                docs.append(info)
            elif PurePosixPath(name).suffix.lower() == ".zip":
                other.append({"path": name, "reason": "nested archive: extract it and upload it separately"})
            elif adapter_for(name) is not None:
                data.append(info)
            else:
                other.append({"path": name, "reason": "not a supported data format (listed only)"})
        return data, docs, other

    def _read(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        from . import adapter_for

        infos, skipped = list_members(path)
        data, doc_infos, other = self._classify(infos)
        wb = frames_workbook(filename, path, sha256_file(path), "zip")
        wb.unsupported += skipped + other
        wb.tree = [{"path": i.filename, "kind": "data" if i in data else ("documentation" if i in doc_infos else "other"),
                    "shape": f"{i.file_size:,} bytes", "depth": i.filename.count("/")} for i in infos]
        if not data:
            refused = "; ".join(f"{s['path']}: {s['reason']}" for s in skipped[:5])
            raise ReadError("The archive contains no file in a supported data format."
                            + (f" Refused: {refused}." if refused else ""))
        tmp = Path(tempfile.mkdtemp(prefix="ilovecsv-zip-"))
        per_member: list[tuple[str, WorkbookData]] = []
        try:
            with zipfile.ZipFile(path) as z:
                for info in data:
                    target = tmp / f"m{len(per_member)}{PurePosixPath(info.filename).suffix.lower()}"
                    with z.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                    member_opts = options.model_copy(update={"tables": None})
                    adapter = adapter_for(info.filename)
                    try:
                        mwb = adapter.load(target, PurePosixPath(info.filename).name, member_opts)
                    except ReadError as e:
                        wb.unsupported.append({"path": info.filename, "reason": f"could not be read: {e}"})
                        continue
                    per_member.append((info.filename, mwb))
                columns = [str(c) for _, m in per_member for s in m.sheets.values() for c in s.df.columns]
                for info in doc_infos:
                    with z.open(info) as src:
                        wb.docs.append(read_doc(info.filename, src.read(2 * 1024 * 1024), columns))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._merge(wb, per_member, options)
        if options.tables:
            missing = [t for t in options.tables if t not in wb.sheets]
            if missing:
                raise ReadError(f"Table(s) not found in the archive: {missing}. Available: {list(wb.sheets)}")
            wb.sheets = {k: v for k, v in wb.sheets.items() if k in options.tables}
        if not wb.sheets:
            raise ReadError("No member of the archive could be turned into a table.")
        return wb

    def _merge(self, wb: WorkbookData, per_member: list[tuple[str, WorkbookData]], options: SourceOptions) -> None:
        """Name tables after their member; merge same-named tables with identical columns if asked."""
        groups: dict[str, list[tuple[str, str, Any, dict]]] = {}
        for member, mwb in per_member:
            stem = PurePosixPath(member).stem
            wb.warnings += [f"{member}: {w}" for w in mwb.warnings]
            wb.unsupported += [{**u, "path": f"{member}:{u['path']}"} for u in mwb.unsupported]
            if mwb.partial:
                wb.partial = wb.partial or {"rows": mwb.partial.get("rows"), "totals": {}}
            single = len(mwb.sheets) == 1
            for tname, sheet in mwb.sheets.items():
                key = tname if not single else stem
                meta = dict(mwb.tables_meta.get(tname, {}))
                total = ((mwb.partial or {}).get("totals") or {}).get(tname)
                if total is None and mwb.partial and single:
                    total = mwb.partial.get("estimated_total_rows")
                meta["total_rows"] = total
                meta["text_values"] = "all" if mwb.file_type == "csv" else meta.get("text_values")
                groups.setdefault(key, []).append((member, tname, sheet, meta))
        for key, items in groups.items():
            same_columns = len({tuple(s.df.columns) for _, _, s, _ in items}) == 1
            if len(items) > 1 and options.combine_tables and same_columns:
                frames = []
                for member, _, sheet, _ in items:
                    df = sheet.df.copy()
                    df.insert(0, "source_file", member)
                    frames.append(df.reset_index(drop=True))
                from ..reader import sheet_from_frame

                merged = sheet_from_frame(safe_table_name(key), pd.concat(frames, ignore_index=True))
                wb.sheets[merged.name] = merged
                totals = [m.get("total_rows") for _, _, _, m in items]
                if wb.partial and all(t is not None for t in totals):
                    wb.partial.setdefault("totals", {})[merged.name] = sum(totals)
                wb.tables_meta[merged.name] = {
                    "source": ", ".join(m for m, _, _, _ in items),
                    "notes": [f"merged {len(items)} same-named tables with identical columns; source_file tells "
                              "them apart"] + items[0][3].get("notes", [])[:3],
                    "members": [m for m, _, _, _ in items], "text_values": items[0][3].get("text_values")}
                continue
            for member, tname, sheet, meta in items:
                # Same name, kept apart: name each after its path in the archive (2023/sensors.csv -> 2023_sensors).
                where = str(PurePosixPath(member).with_suffix(""))
                name = safe_table_name(key if len(items) == 1 else
                                       where if key == PurePosixPath(member).stem else f"{where}_{tname}")
                sheet.name = name
                wb.sheets[name] = sheet
                if wb.partial and meta.get("total_rows") is not None:
                    wb.partial.setdefault("totals", {})[name] = meta["total_rows"]
                wb.tables_meta[name] = {"source": f"{member}" + (f" → {tname}" if tname != key else ""),
                                        "notes": meta.get("notes", []), "members": [member],
                                        "text_values": meta.get("text_values")}

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        return self._read(path, filename, options)

    def inspect(self, path: Path, filename: str, options: SourceOptions) -> Inspection:
        wb = self._read(path, filename, options.model_copy(update={"max_rows": min(options.max_rows or 1000, 1000)}))
        insp = summarise(wb, "zip", path, options)
        totals = (wb.partial or {}).get("totals", {})
        for t in insp.tables:
            if t.name in totals:
                t.rows, t.rows_exact = totals[t.name], True
        return insp
