"""Format adapters: every supported file becomes named tables + metadata through one interface.

    inspect_file(path)   structure first: tables, rows, columns, samples, docs, unsupported parts
    load_dataset(path)   the tables, ready for the shared quality engine
"""

from __future__ import annotations

import struct
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TypeVar

from ..reader import DamagedFileError, ReadError, UnsupportedFormatError, WorkbookData, where
from ..sources import SourceOptions
from .archive import ZipAdapter
from .base import Adapter, Inspection, TableInfo
from .docs import is_doc_file, read_doc
from .scientific import HDF5Adapter, MATAdapter, NumpyAdapter
from .tabular import (
    ARFFAdapter,
    ArrowAdapter,
    DelimitedAdapter,
    ExcelAdapter,
    JSONAdapter,
    ParquetAdapter,
)

ADAPTERS: list[Adapter] = [DelimitedAdapter(), ExcelAdapter(), JSONAdapter(), ParquetAdapter(), ArrowAdapter(),
                           ARFFAdapter(), ZipAdapter(), MATAdapter(), NumpyAdapter(), HDF5Adapter()]
_BY_EXT = {ext: a for a in ADAPTERS for ext in a.extensions}
SUPPORTED_EXTENSIONS = tuple(sorted(_BY_EXT))

# File signatures, so a renamed file is caught before parsing ("text" = must not contain NUL bytes).
SIGNATURES: dict[str, tuple[bytes, ...] | str] = {
    ".xlsx": (b"PK\x03\x04",), ".xlsm": (b"PK\x03\x04",), ".zip": (b"PK\x03\x04", b"PK\x05\x06"),
    ".npz": (b"PK\x03\x04", b"PK\x05\x06"), ".xls": (b"\xd0\xcf\x11\xe0",), ".parquet": (b"PAR1",), ".pq": (b"PAR1",),
    ".feather": (b"ARROW1",), ".arrow": (b"ARROW1",), ".ipc": (b"ARROW1",), ".npy": (b"\x93NUMPY",),
    ".h5": (b"\x89HDF",), ".hdf5": (b"\x89HDF",), ".he5": (b"\x89HDF",),
    ".csv": "text", ".tsv": "text", ".txt": "text", ".json": "text", ".jsonl": "text", ".ndjson": "text",
    ".arff": "text", ".mat": "mat",
}


def adapter_for(filename: str) -> Adapter | None:
    return _BY_EXT.get(PurePosixPath(filename).suffix.lower())


def signature_problem(path: Path, filename: str) -> str | None:
    """Why the file content does not match its extension, or None."""
    ext = PurePosixPath(filename).suffix.lower()
    expected = SIGNATURES.get(ext)
    with open(path, "rb") as f:
        head = f.read(4096)
    if expected is None:
        return f"'{ext}' is not a supported format"
    if expected == "text":
        if b"\x00" in head or head.startswith((b"PK\x03\x04", b"\x89HDF", b"PAR1")):
            return f"the content is binary, not {ext} text"
        return None
    if expected == "mat":  # v5-v7.2 text header, v7.3 HDF5 (text header + HDF5 at 512), or headerless v4
        return None if head.startswith(b"MATLAB") or b"\x89HDF" in head[:1024] or len(head) >= 20 else "empty file"
    if not head.startswith(expected):
        if ext == ".h5" and b"\x89HDF" in head[:1024]:
            return None  # HDF5 with a user block
        return f"the content is not a {ext} file"
    return None


T = TypeVar("T")
# What parsers raise on truncated / corrupted bytes. Anything else is a bug and is not relabelled.
_DAMAGE = (zipfile.BadZipFile, zlib.error, EOFError, OSError, struct.error, UnicodeDecodeError, KeyError,
           IndexError, ValueError)


def _guard(fn: Callable[[], T], filename: str) -> T:
    try:
        return fn()
    except ReadError:
        raise
    except _DAMAGE as e:
        raise DamagedFileError(f"'{filename}' could not be read: it looks damaged or truncated "
                               f"({type(e).__name__}: {e}). Nothing was imported.") from e
    except Exception as e:  # e.g. pyarrow.ArrowInvalid on a truncated Parquet file
        if type(e).__module__.split(".")[0] in ("pyarrow", "h5py", "scipy", "xlrd", "openpyxl"):
            raise DamagedFileError(f"'{filename}' could not be read: it looks damaged or truncated "
                                   f"({type(e).__name__}: {e}). Nothing was imported.") from e
        raise


def _unsupported(filename: str) -> UnsupportedFormatError:
    ext = PurePosixPath(filename).suffix or "(no extension)"
    return UnsupportedFormatError(f"Unsupported file type '{ext}': IloveCSV reads {', '.join(SUPPORTED_EXTENSIONS)}. "
                                  "Nothing was imported.")


def load_dataset(path: str | Path, filename: str | None = None, options: SourceOptions | None = None) -> WorkbookData:
    path = Path(path)
    filename = filename or path.name
    adapter = adapter_for(filename)
    if adapter is None:
        raise _unsupported(filename)
    return _guard(lambda: adapter.load(path, filename, options or SourceOptions()), filename)


def inspect_file(path: str | Path, filename: str | None = None, options: SourceOptions | None = None,
                 sibling_docs: bool = False) -> Inspection:
    """Structure first. `sibling_docs`: also read README / data dictionary files next to the file
    (local use; uploads have no siblings)."""
    path = Path(path)
    filename = filename or path.name
    adapter = adapter_for(filename)
    if adapter is None:
        raise _unsupported(filename)
    # Inspection reports malformed records and damaged members instead of stopping on them.
    opts = (options or SourceOptions()).model_copy(update={"bad_records": "skip"})
    insp = _guard(lambda: adapter.inspect(path, filename, opts), filename)
    if insp.excluded:
        insp.warnings = [w for w in insp.warnings if "PARTIAL: " not in w]
        insp.warnings.append(
            f"{len(insp.excluded)} malformed record(s) or damaged part(s): "
            + "; ".join(f"{where(p)}: {p['problem']}" for p in insp.excluded[:8])
            + ". Prepare stops on them unless you choose to exclude them (bad_records: skip / --skip-bad-records); "
              "excluded records are then listed in the report and the output is marked PARTIAL.")
    if sibling_docs:
        insp.docs += find_sibling_docs(path, [c["name"] for t in insp.tables for c in t.columns])
    return insp


def find_sibling_docs(path: Path, columns: list[str]) -> list[dict]:
    out = []
    for p in sorted(Path(path).parent.iterdir()):
        if p.is_file() and p != path and is_doc_file(p.name) and p.stat().st_size < 5 * 1024 * 1024:
            out.append(read_doc(p.name, p.read_bytes(), columns))
    return out


__all__ = ["ADAPTERS", "SUPPORTED_EXTENSIONS", "Adapter", "Inspection", "TableInfo", "adapter_for", "inspect_file",
           "load_dataset", "signature_problem", "find_sibling_docs"]
