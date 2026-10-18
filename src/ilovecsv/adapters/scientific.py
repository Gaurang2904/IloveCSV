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

import zipfile
import zlib
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..reader import DamagedFileError, ReadError, WorkbookData, note_excluded, sha256_file
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


def _leaf(name: str) -> str:
    return name.replace("/", ".").rsplit(".", 1)[-1] or "value"


def _text_values(a: np.ndarray) -> list[Any]:
    out = []
    for v in a:
        v = _scalar(v) if isinstance(v, np.ndarray) else v
        out.append(v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
    return out


def is_1d(a: Any) -> bool:
    """A plain 1-D array (numbers or text) with more than one element."""
    return (isinstance(a, np.ndarray) and a.ndim == 1 and a.size > 1 and not a.dtype.names
            and a.dtype.kind in "biufcUS")


def one_d_frame(name: str, a: np.ndarray) -> pd.DataFrame:
    if a.dtype.kind in "US":
        return pd.DataFrame({_leaf(name): _text_values(a)})
    return pd.DataFrame(_complex_split(_leaf(name), a))


def align_or_split(arrays: dict[str, np.ndarray], table: str, where: str, options: SourceOptions,
                   tables: dict[str, pd.DataFrame], notes: dict[str, list[str]], warnings: list[str],
                   metadata: list[dict[str, Any]] | None = None) -> dict:
    """Same-length 1-D arrays side by side *may* be the columns of one table, but equal length is the
    only evidence. They are combined only when asked (align_arrays); otherwise each stays its own
    table and the possibility is pointed out. Returns the arrays that were not combined.

    A text array that only lists the names of its sibling arrays (sensor_names = ['voltage', ...])
    describes the others: it becomes metadata, not a table of observations."""
    rest = dict(arrays)
    leaves = {_leaf(k): k for k in arrays}
    described: dict[str, set[str]] = {}
    for k, a in arrays.items():
        if is_1d(a) and a.dtype.kind in "US":
            values = {str(v) for v in _text_values(a)}
            if len(values) >= 2 and values <= set(leaves) - {_leaf(k)}:
                described[_leaf(k)] = values
                rest.pop(k)
                if metadata is not None:
                    metadata.append({"path": k, "reason": f"metadata: lists the names of the arrays "
                                                          f"{', '.join(sorted(values))}; kept in the manifest, "
                                                          "not exported as a table",
                                     "values": sorted(values)})
    by_len: dict[int, list[str]] = {}
    for k, a in rest.items():
        if is_1d(a):
            by_len.setdefault(a.size, []).append(k)
    groups = [(n, keys) for n, keys in by_len.items() if len(keys) >= 2]
    for n, keys in groups:
        names = ", ".join(_leaf(k) for k in keys)
        if not options.align_arrays:
            listed = next((m for m, vals in described.items() if vals == {_leaf(k) for k in keys}), None)
            evidence = (f" '{listed}' lists exactly these arrays, which supports that they belong together."
                        if listed else " Equal length alone does not prove it, so")
            warnings.append(f"{where}: {names} each have {n} elements. They may be the columns of one table "
                            f"(element k of each = one record).{evidence}"
                            + (" They are" if listed else " they are")
                            + " kept as separate tables. If they are aligned, prepare with align_arrays "
                              "(CLI --align-arrays).")
            continue
        name = table if len(groups) == 1 else f"{table}_{n}"
        cols: dict[str, Any] = {}
        for k in keys:
            cols.update(one_d_frame(k, arrays[k]).to_dict("list"))
            rest.pop(k)
        tables[name] = pd.DataFrame(cols)
        notes[name] = [f"aligned on request (align_arrays): {names} combined as columns, element k of each "
                       "array treated as one record"]
    return rest


def generic_tables(name: str, obj: Any, tables: dict[str, pd.DataFrame], notes: dict[str, list[str]],
                   unsupported: list[dict[str, Any]], depth: int = 0, warnings: list[str] | None = None,
                   context: list[str] | None = None) -> None:
    """Find the parts of a structure that have exactly one obvious table form. `context`: numeric
    per-element fields to repeat on every sample row of a struct array's series table."""
    warnings = [] if warnings is None else warnings
    if isinstance(obj, np.ndarray) and obj.dtype.names:  # structured array
        flat = obj.reshape(-1)
        tables[name] = pd.DataFrame({f: list(flat[f]) for f in obj.dtype.names})
        notes[name] = ["structured array: one column per field"]
    elif _is_numeric_array(obj):
        if obj.ndim == 1:
            tables[name] = one_d_frame(name, obj)
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
            generic_tables(f"{name}.{k}", v, tables, notes, unsupported, depth + 1, warnings, context)
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        _struct_array(name, obj, tables, notes, unsupported, warnings, context or [])
    elif isinstance(obj, list):
        unsupported.append({"path": name, "reason": "cell array of mixed content: no safe table form"})
    elif isinstance(obj, np.ndarray) and obj.ndim == 1 and obj.dtype.kind in "USO" and obj.size > 1:
        tables[name] = pd.DataFrame({_leaf(name): _text_values(obj)})
        notes[name] = ["1-D text array: one row per element"]
    elif np.isscalar(obj) or (isinstance(obj, np.ndarray) and obj.size <= 1):
        if depth == 0:  # nested scalars are reported as metadata on their parent table
            unsupported.append({"path": name, "reason": f"single value ({_scalar(obj)!r}): metadata, not a table"})
    else:
        unsupported.append({"path": name, "reason": f"{type(obj).__name__}: no safe table form"})


def _struct_array(name: str, obj: list[dict], tables: dict[str, pd.DataFrame], notes: dict[str, list[str]],
                  unsupported: list[dict[str, Any]], warnings: list[str], context: list[str]) -> None:
    """Struct array (e.g. battery.cycle(1..n)): scalar fields -> one row per element; per-element 1-D
    series -> one long table (element index, sample index, one column per series), so element
    identity is kept and nothing is squeezed into one cell."""
    fields = list(dict.fromkeys(f for x in obj for f in x))

    def is_scalar(v: Any) -> bool:
        return v is None or np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1)

    def is_series(v: Any) -> bool:
        return v is None or (isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind in "biufc") or is_scalar(v)

    scalar_fields = [f for f in fields if all(is_scalar(x.get(f)) for x in obj)]
    series_fields = [f for f in fields if f not in scalar_fields and all(is_series(x.get(f)) for x in obj)]
    other = [f for f in fields if f not in scalar_fields and f not in series_fields]
    index = f"{_leaf(name)}_index"
    # Repeated on every sample row so the series table can be used without a join: descriptive text
    # fields (cycle type) always; numeric fields only when named, because a per-element number such
    # as Capacity is often the quantity to predict and must not be copied onto every sample by default.
    text_fields = [f for f in scalar_fields if all(x.get(f) is None or isinstance(_scalar(x.get(f)), str) for x in obj)]
    ctx = [f for f in scalar_fields if f in text_fields or f in context]
    if scalar_fields:
        rows = [{index: i, **{f: _scalar(x.get(f)) for f in scalar_fields}} for i, x in enumerate(obj, start=1)]
        tables[name] = pd.DataFrame(rows)
        notes[name] = [f"struct array: one row per element ({len(obj)}), scalar fields as columns; "
                       f"{index} numbers the elements in file order"]
    if series_fields:
        frames = []
        for i, x in enumerate(obj, start=1):
            series = {f: np.atleast_1d(np.asarray(x[f])) for f in series_fields if x.get(f) is not None}
            lengths = {f: a.size for f, a in series.items()}
            n = max(lengths.values(), default=0)
            if len(set(lengths.values())) > 1:
                detail = ", ".join(f"{f} {m}" for f, m in lengths.items())
                warnings.append(f"{name}({i}): series lengths differ ({detail}); the shorter ones are padded with "
                                "missing values at the end. Check that the samples line up before using them "
                                "together.")
            frame: dict[str, Any] = {index: np.full(n, i), **{f: [_scalar(x.get(f))] * n for f in ctx},
                                     "sample_index": np.arange(n)}
            for f, a in series.items():
                for col, part in _complex_split(f, a).items():
                    padded = np.full(n, np.nan)
                    padded[: part.size] = part
                    frame[col] = padded
            frames.append(pd.DataFrame(frame))
        long_name = f"{name}.series"
        tables[long_name] = pd.concat(frames, ignore_index=True)
        notes[long_name] = [f"per-element series ({', '.join(series_fields)}) in long format: one row per sample; "
                            f"{index} links each sample to its row in {name}"]
        skipped = [f for f in scalar_fields if f not in ctx]
        if ctx:
            notes[long_name].append(f"copied from {name} onto every sample row: {', '.join(ctx)}")
        if skipped:
            notes[long_name].append(f"not copied (numeric per-element values, possibly an outcome such as capacity): "
                                    f"{', '.join(skipped)}; join on {index}, or list them in series_context")
    if other:
        unsupported.append({"path": f"{name}[].({', '.join(other)})",
                            "reason": "fields hold multi-dimensional arrays or nested structures; no safe "
                                      "single-table form (a layout-specific adapter is needed)"})


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
            frame = {"battery": battery, "cycle_index": idx, "type": ctype, "ambient_temperature": row["ambient_temperature"],
                     "sample_index": np.arange(n)}
            for k, a in series.items():
                col = "time_s" if k == "Time" else k.lower()
                padded = np.full(n, np.nan)
                padded[: a.size] = a.real if np.iscomplexobj(a) else a
                frame[col] = padded
            meas.append(pd.DataFrame(frame))
        elif ctype == "impedance" and series:
            frame = {"battery": battery, "cycle_index": idx, "ambient_temperature": row["ambient_temperature"],
                     "sample_index": np.arange(n)}
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
                    "(battery, cycle_index, sample_index), with the cycle type and ambient_temperature repeated "
                    "on each row; current_charge/voltage_charge only on charge cycles, "
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
            raise DamagedFileError(f"Not a readable MATLAB file: it is damaged, truncated or not a .mat file ({e}). "
                                   "Nothing was imported.") from e
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
        variables = align_or_split(variables, Path(filename).stem, filename, options, tables, notes, wb.warnings,
                               metadata=wb.unsupported)
        for k, v in variables.items():
            generic_tables(k, v, tables, notes, wb.unsupported, warnings=wb.warnings, context=options.series_context)
        for t, df in tables.items():
            if options.tables and t not in options.tables:
                continue
            add_frame(wb, t, df, options, source=f"{filename} → {t}", notes=notes.get(t, []))
        if not wb.sheets:
            wb.warnings.append("No part of this file has a safe table form; see the structure and the reasons.")
        return wb


def _attrs(attrs: Any) -> dict[str, str]:
    """HDF5 attributes (units, descriptions, provenance) as short strings for the structure view."""
    out = {}
    for k in list(attrs)[:12]:
        v = attrs[k]
        v = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
        out[str(k)] = str(v.tolist() if isinstance(v, np.ndarray) else v)[:120]
    return out


def _raise(e: Exception):
    raise ReadError(f"Not a readable MATLAB file: {e}") from e


# ----------------------------------------------------------------------------- NumPy


class NumpyAdapter(Adapter):
    name = "numpy"
    extensions = (".npy", ".npz")

    def load(self, path: Path, filename: str, options: SourceOptions) -> WorkbookData:
        suffix = Path(filename).suffix.lower()
        wb = frames_workbook(filename, path, sha256_file(path), suffix.lstrip("."))
        if suffix == ".npz":  # an .npz is a zip: same decompression limits as archives
            from .archive import MAX_RATIO, MAX_TOTAL_BYTES

            try:
                with zipfile.ZipFile(path) as z:
                    infos = z.infolist()
            except (zipfile.BadZipFile, EOFError, OSError) as e:
                raise DamagedFileError(f"The .npz file is damaged or truncated: it is not a readable NumPy archive "
                                       f"({e}). No arrays were loaded.") from e
            total = sum(i.file_size for i in infos)
            if total > MAX_TOTAL_BYTES or any(i.compress_size and i.file_size > 1 << 20
                                              and i.file_size / i.compress_size > MAX_RATIO for i in infos):
                raise ReadError(f"The .npz expands to {total / 1e9:.1f} GB or has a suspicious compression ratio; "
                                "refused (decompression limit). Nothing was imported.")
        try:
            loaded = np.load(path, allow_pickle=False)  # never unpickle untrusted files
        except ValueError as e:
            if "pickle" in str(e).lower():
                raise ReadError("This NumPy file contains Python objects that need pickle to load; IloveCSV does "
                                "not unpickle untrusted files. Re-save it with numeric/structured arrays.") from e
            raise DamagedFileError(f"The {suffix} file is damaged or truncated ({e}). No arrays were loaded.") from e
        except (zipfile.BadZipFile, EOFError, OSError) as e:
            raise DamagedFileError(f"The {suffix} file is damaged or truncated: it is not a readable NumPy "
                                   f"{'archive' if suffix == '.npz' else 'array'} ({e}). No arrays were loaded.") from e
        if suffix == ".npy":
            arrays = {Path(filename).stem: loaded}
        else:
            arrays, damaged = {}, []
            for k in loaded.files:  # arrays are read lazily: a damaged member fails here
                try:
                    arrays[k] = loaded[k]
                except ValueError as e:
                    if "pickle" in str(e).lower():
                        wb.unsupported.append({"path": k, "reason": "holds Python objects (needs pickle): not "
                                                                    "loaded, IloveCSV does not unpickle untrusted data"})
                        continue
                    damaged.append({"member": k, "line": None, "problem": f"damaged array, not loaded ({e})", "raw": ""})
                except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as e:
                    damaged.append({"member": k, "line": None, "problem": f"damaged array, not loaded ({e})", "raw": ""})
            if damaged and options.bad_records != "skip":
                raise DamagedFileError(
                    f"{len(damaged)} of {len(loaded.files)} arrays in the .npz are damaged: "
                    + "; ".join(f"{d['member']}: {d['problem']}" for d in damaged[:5])
                    + f". Intact: {', '.join(arrays) or 'none'}. Nothing was imported; re-run with bad_records: "
                      "skip (CLI --skip-bad-records) to load only the intact arrays (output marked PARTIAL).",
                    problems=damaged)
            note_excluded(wb, damaged, "damaged array(s)")
        for k, a in arrays.items():
            wb.tree.append({"path": k, "kind": f"array {a.dtype}", "shape": _shape(a), "depth": 0})
        tables: dict[str, pd.DataFrame] = {}
        notes: dict[str, list[str]] = {}
        arrays = align_or_split(arrays, Path(filename).stem, filename, options, tables, notes, wb.warnings,
                               metadata=wb.unsupported)
        for k, a in arrays.items():
            generic_tables(k, a, tables, notes, wb.unsupported, warnings=wb.warnings)
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
            raise DamagedFileError(f"Not a readable HDF5 file: it is damaged, truncated or not HDF5 ({e}). Nothing was "
                                   "imported.") from e
        wb = frames_workbook(filename, path, sha256_file(path), "mat_v7.3" if matlab else "hdf5")
        tables: dict[str, pd.DataFrame] = {}
        notes: dict[str, list[str]] = {}
        one_d: dict[str, dict[str, np.ndarray]] = {}  # parent group -> 1-D datasets, aligned only on request
        with f:
            if any("pandas_type" in g.attrs for g in f.values() if hasattr(g, "attrs")):
                wb.warnings.append("This looks like a pandas HDFStore; pandas.read_hdf reconstructs it exactly "
                                   "(needs PyTables). The generic reader shows the raw arrays.")
            if f.attrs and not matlab:
                wb.tree.append({"path": "/", "kind": "file", "shape": "-", "depth": 0, "attrs": _attrs(f.attrs)})

            def visit(name: str, node: Any) -> None:
                if len(wb.tree) < MAX_TREE:
                    kind = "group" if isinstance(node, h5py.Group) else f"dataset {node.dtype}"
                    entry = {"path": name, "kind": kind, "shape": "x".join(map(str, getattr(node, "shape", ())))
                             or "-", "depth": name.count("/")}
                    if node.attrs and not matlab:
                        entry["attrs"] = _attrs(node.attrs)
                    wb.tree.append(entry)
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
                if is_1d(arr):
                    parent = name.rsplit("/", 1)[0] if "/" in name else ""
                    one_d.setdefault(parent, {})[name.replace("/", ".")] = arr
                    return
                generic_tables(name.replace("/", "."), arr, tables, notes, wb.unsupported, warnings=wb.warnings)

            f.visititems(visit)
        for parent, arrays in one_d.items():
            table = parent.replace("/", ".") or Path(filename).stem
            rest = align_or_split(arrays, table, f"{filename} /{parent}", options, tables, notes, wb.warnings,
                               metadata=wb.unsupported)
            for k, a in rest.items():
                generic_tables(k, a, tables, notes, wb.unsupported, warnings=wb.warnings)
        for t, df in tables.items():
            if not options.tables or t in options.tables:
                add_frame(wb, t, df, options, source=f"{filename} → /{t.replace('.', '/')}", notes=notes.get(t, []))
        if not wb.sheets:
            wb.warnings.append("No dataset in this file has a safe table form; see the structure and the reasons.")
        return wb
