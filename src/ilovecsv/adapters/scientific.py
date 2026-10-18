"""Scientific formats: MATLAB .mat (v4-v7.2 via SciPy, v7.3 via h5py), NumPy .npy/.npz, HDF5.

Arrays are not automatically tables. The generic rules only accept shapes with one obvious
reading, and record everything else under `unsupported` with the reason:

  2-D numeric array (n x k)                -> table, columns c0..c{k-1} (the source has no names)
  1-D arrays of equal length in one struct -> table, one column per field
  struct array with scalar fields          -> table, one row per element
  structured / compound array              -> table, one column per field
  >2-D arrays, cell arrays, mixed objects  -> unsupported ("readable, but no safe row layout")

Known layouts get a dedicated converter instead of a guess. NASA's Li-ion battery aging files
(B0005.mat ...: cycles of charge / discharge / impedance, each holding variable-length series)
become three tables: one row per cycle, the charge/discharge measurements in long format, and
the impedance spectra, so no time series is squeezed into one awkward sheet.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..reader import ReadError, WorkbookData, sha256_file
from ..sources import SourceOptions
from .base import Adapter, frames_workbook
from .tabular import add_frame

MAX_TREE = 400  # nodes listed in a structure preview


def _is_numeric_array(a: Any) -> bool:
    return isinstance(a, np.ndarray) and a.dtype.kind in "biufc" and a.size > 0


def _shape(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return "x".join(map(str, v.shape)) or "scalar"
    if isinstance(v, list):
        return f"list[{len(v)}]"
    return "scalar"


def _kind(v: Any) -> str:
    if isinstance(v, dict):
        return "struct"
    if isinstance(v, list):
        return "struct array / cell" if v and all(isinstance(x, dict) for x in v[:20]) else "cell array"
    if isinstance(v, np.ndarray):
        return f"array {v.dtype}"
    return type(v).__name__


def _tree(obj: Any, path: str, out: list[dict[str, Any]], depth: int = 0) -> None:
    if len(out) >= MAX_TREE:
        return
    out.append({"path": path, "kind": _kind(obj), "shape": _shape(obj), "depth": depth})
    if isinstance(obj, dict) and depth < 4:
        for k, v in obj.items():
            _tree(v, f"{path}.{k}", out, depth + 1)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and depth < 4:
        _tree(obj[0], f"{path}[0]", out, depth + 1)  # first element shows the element layout


def _complex_split(name: str, a: np.ndarray) -> dict[str, np.ndarray]:
    if np.iscomplexobj(a):
        return {f"{name}_re": a.real, f"{name}_im": a.imag}
    return {name: a}


# ----------------------------------------------------------------------------- generic conversion


def generic_tables(name: str, obj: Any, tables: dict[str, pd.DataFrame], notes: dict[str, list[str]],
                   unsupported: list[dict[str, Any]], depth: int = 0) -> None:
    """Find the parts of a structure that have exactly one obvious table form."""
    if isinstance(obj, np.ndarray) and obj.dtype.names:  # structured array
        flat = obj.reshape(-1)
        tables[name] = pd.DataFrame({f: list(flat[f]) for f in obj.dtype.names})
        notes[name] = ["structured array: one column per field"]
    elif _is_numeric_array(obj):
        if obj.ndim == 1:
            tables[name] = pd.DataFrame(_complex_split("value", obj))
            notes[name] = ["1-D array: one row per element"]
        elif obj.ndim == 2:
            cols: dict[str, np.ndarray] = {}
            for j in range(obj.shape[1]):
                cols.update(_complex_split(f"c{j}", obj[:, j]))
            tables[name] = pd.DataFrame(cols)
            notes[name] = [f"2-D array {obj.shape[0]}x{obj.shape[1]}: rows as rows, columns named c0.. "
                           "(the source has no column names; check the documentation)"]
        else:
            unsupported.append({"path": name, "reason": f"{obj.ndim}-D array {obj.shape}: readable, but there is no "
                                                        "safe row layout (choose an axis in your own code)"})
    elif isinstance(obj, dict) and depth < 4:
        arrays = {k: np.atleast_1d(v) for k, v in obj.items() if _is_numeric_array(np.asarray(v))
                  and np.asarray(v).ndim <= 1 and np.asarray(v).size > 1}
        lengths = Counter(a.size for a in arrays.values())
        if arrays and len(lengths) == 1 and len(arrays) >= 2:
            cols: dict[str, np.ndarray] = {}
            for k, a in arrays.items():
                cols.update(_complex_split(k, a))
            tables[name] = pd.DataFrame(cols)
            notes[name] = [f"fields with equal-length 1-D arrays ({', '.join(arrays)}) became columns"]
            rest = {k: v for k, v in obj.items() if k not in arrays}
        else:
            rest = obj
        scalars = {k: v for k, v in rest.items() if np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1)}
        if scalars and name in tables:
            notes[name].append("scalar fields kept as metadata: " + ", ".join(f"{k}={_scalar(v)}" for k, v in
                                                                             list(scalars.items())[:10]))
        for k, v in rest.items():
            if k in scalars:
                continue
            generic_tables(f"{name}.{k}", v, tables, notes, unsupported, depth + 1)
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        fields = list(obj[0])
        scalar_fields = [f for f in fields if all(np.isscalar(x.get(f)) or
                                                  (isinstance(x.get(f), np.ndarray) and x.get(f).size == 1)
                                                  or x.get(f) is None for x in obj)]
        if scalar_fields:
            tables[name] = pd.DataFrame([{f: _scalar(x.get(f)) for f in scalar_fields} for x in obj])
            notes[name] = [f"struct array: one row per element ({len(obj)}), scalar fields as columns"]
            dropped = [f for f in fields if f not in scalar_fields]
            if dropped:
                unsupported.append({"path": f"{name}[].({', '.join(dropped)})",
                                    "reason": "fields hold arrays or nested structures of varying size; no safe "
                                              "single-table form (a layout-specific adapter is needed)"})
        else:
            unsupported.append({"path": name, "reason": "struct array whose fields are all arrays or structures; "
                                                        "no safe single-table form without a known layout"})
    elif isinstance(obj, list):
        unsupported.append({"path": name, "reason": "cell array of mixed content: no safe table form"})
    elif isinstance(obj, np.ndarray) and obj.ndim == 1 and obj.dtype.kind in "USO" and obj.size > 1:
        tables[name] = pd.DataFrame({"value": [_scalar(v) if isinstance(v, np.ndarray) else v for v in obj]})
        notes[name] = ["1-D text array: one row per element"]
    elif np.isscalar(obj) or (isinstance(obj, np.ndarray) and obj.size <= 1):
        if depth == 0:  # nested scalars are reported as metadata on their parent table
            unsupported.append({"path": name, "reason": f"single value ({_scalar(obj)!r}): metadata, not a table"})
    else:
        unsupported.append({"path": name, "reason": f"{type(obj).__name__}: no safe table form"})


def _scalar(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else None
    return v


# ----------------------------------------------------------------------------- NASA battery layout


def is_nasa_battery(var: Any) -> bool:
    if not isinstance(var, dict) or "cycle" not in var:
        return False
    cycles = var["cycle"]
    cycles = cycles if isinstance(cycles, list) else [cycles]
    return bool(cycles) and all(isinstance(c, dict) and {"type", "data"} <= set(c) for c in cycles[:20])


def _matlab_clock(v: Any) -> datetime | None:
    a = np.atleast_1d(np.asarray(v, dtype=float))
    if a.size < 6 or not np.all(np.isfinite(a[:6])):
        return None
    y, mo, d, h, mi, s = a[:6]
    try:
        return datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=float(s))
    except ValueError:
        return None


def nasa_battery_tables(battery: str, var: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    cycles = var["cycle"] if isinstance(var["cycle"], list) else [var["cycle"]]
    rows, meas, imp, notes = [], [], [], []
    for idx, c in enumerate(cycles, start=1):
        data = c.get("data") or {}
        ctype = str(c.get("type", "")).strip()
        series = {k: np.atleast_1d(np.asarray(v)) for k, v in data.items()
                  if isinstance(v, np.ndarray) and v.size > 1}
        n = max((a.size for a in series.values()), default=0)
        row = {"battery": battery, "cycle_index": idx, "type": ctype,
               "ambient_temperature": _scalar(np.asarray(c.get("ambient_temperature"))) if c.get(
                   "ambient_temperature") is not None else None,
               "start_time": _matlab_clock(c.get("time")) if c.get("time") is not None else None,
               "n_samples": n}
        for k, v in data.items():  # scalar results: Capacity (discharge), Re / Rct (impedance)
            if not (isinstance(v, np.ndarray) and v.size > 1):
                val = _scalar(np.asarray(v))
                row[k.lower()] = val.real if isinstance(val, complex) else val
        rows.append(row)
        if ctype in ("charge", "discharge") and series:
            lengths = {k: a.size for k, a in series.items()}
            if len(set(lengths.values())) > 1:
                notes.append(f"cycle {idx}: series of different lengths {lengths}; shorter ones padded with "
                             "missing values")
            frame = {"battery": battery, "cycle_index": idx, "type": ctype, "sample_index": np.arange(n)}
            for k, a in series.items():
                col = "time_s" if k == "Time" else k.lower()
                padded = np.full(n, np.nan)
                padded[: a.size] = a.real if np.iscomplexobj(a) else a
                frame[col] = padded
            meas.append(pd.DataFrame(frame))
        elif ctype == "impedance" and series:
            frame = {"battery": battery, "cycle_index": idx, "sample_index": np.arange(n)}
            for k, a in series.items():
                for col, part in _complex_split(k.lower(), a.astype(complex) if np.iscomplexobj(a) else a).items():
                    padded = np.full(n, np.nan)
                    padded[: part.size] = part
                    frame[col] = padded
            imp.append(pd.DataFrame(frame))
    tables = {"cycles": pd.DataFrame(rows)}
    if meas:
        tables["measurements"] = pd.concat(meas, ignore_index=True)
    if imp:
        tables["impedance"] = pd.concat(imp, ignore_index=True)
    return tables, notes


NASA_NOTES = {
    "cycles": "one row per cycle in file order (cycle_index 1..n); capacity only on discharge cycles, "
              "re / rct only on impedance cycles; start_time from MATLAB clock vectors",
    "measurements": "charge and discharge time series in long format: one row per sample "
                    "(battery, cycle_index, sample_index); current_charge/voltage_charge only on charge cycles, "
                    "current_load/voltage_load only on discharge cycles",
    "impedance": "impedance spectra, one row per frequency sample; complex values split into _re/_im",
}


# ----------------------------------------------------------------------------- MATLAB


class MATAdapter(Adapter):
    name = "mat"
    extensions = (".mat",)

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        with open(path, "rb") as f:
            head = f.read(128)
        if b"MATLAB 7.3" in head:
            return HDF5Adapter().load(path, filename, options, matlab=True)
        from scipy.io import loadmat

        try:
            content = loadmat(str(path), simplify_cells=True)
        except NotImplementedError as e:  # v7.3 without the text header
            return HDF5Adapter().load(path, filename, options, matlab=True) if "HDF" in str(e) else _raise(e)
        except Exception as e:
            raise ReadError(f"Not a readable MATLAB file: {e}") from e
        variables = {k: v for k, v in content.items() if not k.startswith("__")}
        wb = frames_workbook(filename, path, sha256_file(path), "mat")
        for k, v in variables.items():
            _tree(v, k, wb.tree)
        nasa = {k: v for k, v in variables.items() if is_nasa_battery(v)}
        if nasa and options.mat_layout in ("auto", "nasa_battery"):
            wb.source_format = "nasa_battery_mat"
            collected: dict[str, list[pd.DataFrame]] = {}
            extra_notes: list[str] = []
            for battery, var in nasa.items():
                tables, notes = nasa_battery_tables(battery, var)
                extra_notes += notes
                for t, df in tables.items():
                    collected.setdefault(t, []).append(df)
            for t, frames in collected.items():
                if options.tables and t not in options.tables:
                    continue
                add_frame(wb, t, pd.concat(frames, ignore_index=True), options,
                          source=f"{filename} (NASA battery layout: {', '.join(nasa)})",
                          notes=[NASA_NOTES.get(t, "")] + extra_notes[:5])
            others = [k for k in variables if k not in nasa]
            if others:
                wb.warnings.append(f"Variables outside the battery layout were not converted: {others}")
            return wb
        if options.mat_layout == "nasa_battery":
            raise ReadError("mat_layout is nasa_battery, but the file does not have the NASA battery structure.")
        tables: dict[str, pd.DataFrame] = {}
        notes: dict[str, list[str]] = {}
        for k, v in variables.items():
            generic_tables(k, v, tables, notes, wb.unsupported)
        for t, df in tables.items():
            if options.tables and t not in options.tables:
                continue
            add_frame(wb, t, df, options, source=f"{filename} → {t}", notes=notes.get(t, []))
        if not wb.sheets:
            wb.warnings.append("No part of this file has a safe table form; see the structure and the reasons.")
        return wb


def _raise(e: Exception):
    raise ReadError(f"Not a readable MATLAB file: {e}") from e


# ----------------------------------------------------------------------------- NumPy


class NumpyAdapter(Adapter):
    name = "numpy"
    extensions = (".npy", ".npz")

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        suffix = Path(filename).suffix.lower()
        wb = frames_workbook(filename, path, sha256_file(path), suffix.lstrip("."))
        try:
            loaded = np.load(path, allow_pickle=False)  # never unpickle untrusted files
        except ValueError as e:
            if "pickle" in str(e).lower():
                raise ReadError("This NumPy file contains Python objects that need pickle to load; IloveCSV does "
                                "not unpickle untrusted files. Re-save it with numeric/structured arrays.") from e
            raise ReadError(f"Not a readable NumPy file: {e}") from e
        arrays = {Path(filename).stem: loaded} if suffix == ".npy" else {k: loaded[k] for k in loaded.files}
        for k, a in arrays.items():
            wb.tree.append({"path": k, "kind": f"array {a.dtype}", "shape": _shape(a), "depth": 0})
        one_d = {k: a for k, a in arrays.items() if isinstance(a, np.ndarray) and a.ndim == 1 and not a.dtype.names}
        if suffix == ".npz" and len(one_d) == len(arrays) > 1 and len({a.size for a in one_d.values()}) == 1:
            cols: dict[str, np.ndarray] = {}
            for k, a in one_d.items():
                cols.update(_complex_split(k, a))
            add_frame(wb, Path(filename).stem, pd.DataFrame(cols), options, source=filename,
                      notes=[f"equal-length 1-D arrays ({', '.join(one_d)}) became columns"])
            return wb
        tables: dict[str, pd.DataFrame] = {}
        notes: dict[str, list[str]] = {}
        for k, a in arrays.items():
            generic_tables(k, a, tables, notes, wb.unsupported)
        for t, df in tables.items():
            if not options.tables or t in options.tables:
                add_frame(wb, t, df, options, source=f"{filename} → {t}", notes=notes.get(t, []))
        if not wb.sheets:
            wb.warnings.append("No array in this file has a safe table form.")
        return wb


# ----------------------------------------------------------------------------- HDF5 (and MATLAB v7.3)


class HDF5Adapter(Adapter):
    name = "hdf5"
    extensions = (".h5", ".hdf5", ".he5")

    def load(self, path: Path, filename: str, options: SourceOptions, matlab: bool = False) -> WorkbookData:
        try:
            import h5py
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ReadError("HDF5 and MATLAB v7.3 files need h5py: pip install 'ilovecsv[scientific]'") from e
        try:
            f = h5py.File(path, "r")
        except Exception as e:
            raise ReadError(f"Not a readable HDF5 file: {e}") from e
        wb = frames_workbook(filename, path, sha256_file(path), "mat_v7.3" if matlab else "hdf5")
        tables: dict[str, pd.DataFrame] = {}
        notes: dict[str, list[str]] = {}
        with f:
            if any("pandas_type" in g.attrs for g in f.values() if hasattr(g, "attrs")):
                wb.warnings.append("This looks like a pandas HDFStore; pandas.read_hdf reconstructs it exactly "
                                   "(needs PyTables). The generic reader shows the raw arrays.")

            def visit(name: str, node: Any) -> None:
                if len(wb.tree) < MAX_TREE:
                    kind = "group" if isinstance(node, h5py.Group) else f"dataset {node.dtype}"
                    wb.tree.append({"path": name, "kind": kind, "shape": "x".join(map(str, getattr(node, "shape", ())))
                                    or "-", "depth": name.count("/")})
                if not isinstance(node, h5py.Dataset) or name.startswith("#refs#"):
                    return
                if node.dtype.kind == "O" or node.dtype.kind == "V" and not node.dtype.names:
                    wb.unsupported.append({"path": name, "reason": "object references / opaque data (e.g. a MATLAB "
                                                                   "cell array): no safe table form"})
                    return
                if matlab and node.attrs.get("MATLAB_class", b"") in (b"char",):
                    return  # strings stored as uint16 code points: metadata, not data
                rows = node.shape[0] if node.ndim else 1
                if options.max_rows is None and node.ndim <= 2 and rows > options.row_limit:
                    wb.unsupported.append({"path": name, "reason": f"{rows:,} rows exceeds the processing limit; "
                                                                   "set max_rows to sample it"})
                    return
                data = node[: options.max_rows] if (options.max_rows and node.ndim) else node[()]
                arr = np.asarray(data)
                if matlab and arr.ndim == 2:
                    arr = arr.T  # MATLAB stores column-major
                generic_tables(name.replace("/", "."), arr, tables, notes, wb.unsupported)

            f.visititems(visit)
        for t, df in tables.items():
            if not options.tables or t in options.tables:
                add_frame(wb, t, df, options, source=f"{filename} → /{t.replace('.', '/')}", notes=notes.get(t, []))
        if not wb.sheets:
            wb.warnings.append("No dataset in this file has a safe table form; see the structure and the reasons.")
        return wb
