"""The hand-off to an ML pipeline: feature schema, target summary, leakage suspects, typed data.

IloveCSV stops at data quality. Nothing here learns parameters from the data to transform it
(no imputation, scaling or encoding): those must be fitted on the training split only, inside the
user's own pipeline, or information from validation/test rows leaks into training.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .profiling import ColumnProfile, SheetProfile, collapse_ws
from .values import normalise_nulls, to_bool, to_date, to_number, to_text

if TYPE_CHECKING:
    from .pipeline import RunResult
    from .reader import SheetData
    from .rules.schema import SheetRules

ROLES = ("numeric", "categorical", "datetime", "boolean", "text", "identifier", "pii", "target", "empty")
FEATURE_ROLES = ("numeric", "categorical", "datetime", "boolean", "text")
_DTYPE = {"int": "Int64", "float": "Float64", "date": "datetime64[ns]", "bool": "boolean", "category": "category"}


def column_role(name: str, c: ColumnProfile, target: str | None) -> str:
    if target is not None and name == target:
        return "target"
    if c.inferred_type == "empty":
        return "empty"
    if c.semantic in ("email", "phone"):
        return "pii"
    if c.is_identifier or c.sequential_id:
        return "identifier"
    return {"int": "numeric", "float": "numeric", "date": "datetime", "bool": "boolean",
            "category": "categorical"}.get(c.inferred_type, "text")


def column_dtype(c: ColumnProfile, role: str) -> str:
    if role in ("identifier", "pii", "text", "empty"):
        return "string"
    return _DTYPE.get(c.inferred_type, "string")


def sheet_schema(prof: SheetProfile, srules: SheetRules | None) -> dict[str, Any]:
    target = srules.target if srules else None
    columns = []
    groups: dict[str, list[str]] = defaultdict(list)
    for name, c in prof.columns.items():
        role = column_role(str(name), c, target)
        groups[role].append(str(name))
        columns.append({
            "name": str(name), "role": role, "dtype": column_dtype(c, role), "inferred_type": c.inferred_type,
            "missing": c.n_missing, "missing_rate": c.missing_rate, "distinct": c.n_distinct,
        })
    return {
        "sheet": prof.name,
        "columns": columns,
        "features": {r: groups.get(r, []) for r in FEATURE_ROLES},
        "excluded": {"identifier": groups.get("identifier", []), "pii": groups.get("pii", []),
                     "empty": groups.get("empty", []), "target": target},
        "high_missing": [col["name"] for col in columns if col["missing_rate"] > 0.5],
    }


def feature_schema(run: RunResult) -> list[dict[str, Any]]:
    """One schema per validated sheet, from the profiles of `run` (pass the after-cleaning run)."""
    out = []
    for srules in run.rules.sheets:
        name = srules.sheet or run.wb.first_sheet.name
        if name in run.profiles:
            out.append(sheet_schema(run.profiles[name], srules))
    return out


# ----------------------------------------------------------------------------- target


def _norm(v: Any, nulls) -> str | None:
    t = to_text(v, nulls)
    return None if t is None else collapse_ws(t)


def target_summary(sheet: SheetData, target: str, nulls) -> dict[str, Any] | None:
    if target not in sheet.df.columns:
        return None
    values = [_norm(v, nulls) for v in sheet.df[target]]
    present = [v for v in values if v is not None]
    summary: dict[str, Any] = {"column": target, "rows": len(values), "missing": len(values) - len(present)}
    nums = [to_number(v, nulls) for v in present]
    distinct = set(present)
    if present and all(n is not None for n in nums) and len(distinct) > 20:
        s = pd.Series(nums, dtype="float64")
        summary.update({"kind": "regression", "min": float(s.min()), "median": float(s.median()),
                        "max": float(s.max())})
    else:
        counts = Counter(present)
        total = sum(counts.values()) or 1
        summary.update({"kind": "classification", "classes": [
            {"label": k, "count": n, "share": round(n / total, 4)} for k, n in counts.most_common()]})
        if len(counts) >= 2:
            summary["imbalance_ratio"] = round(counts.most_common()[0][1] / counts.most_common()[-1][1], 2)
    return summary


LEAKAGE_ACCURACY = 0.99  # near-perfect: one stray data error should not hide a leak


def _separation(pairs: list[tuple[Any, str]]) -> tuple[float, dict[Any, str]]:
    """Accuracy of predicting the label from the key alone (majority label per key)."""
    groups: dict[Any, Counter] = defaultdict(Counter)
    for k, y in pairs:
        groups[k][y] += 1
    majority = {k: c.most_common(1)[0][0] for k, c in groups.items()}
    hits = sum(c.most_common(1)[0][1] for c in groups.values())
    return hits / len(pairs), majority


def leakage_suspects(sheet: SheetData, prof: SheetProfile, target: str, nulls) -> list[dict[str, Any]]:
    """Columns that predict a categorical target (near-)perfectly: review before using them.

    * missingness: whether the column is empty predicts the label (e.g. cancellation_date exists
      only for churned customers);
    * value: the column's value predicts the label (e.g. a status set after the outcome).
    Prediction accuracy >= 99% counts, so a single data error does not hide a leak.
    """
    df = sheet.df
    if target not in df.columns:
        return []
    labels = [_norm(v, nulls) for v in df[target]]
    if len({v for v in labels if v is not None}) < 2 or len({v for v in labels if v is not None}) > 20:
        return []  # only for classification targets
    out = []
    for col in df.columns:
        c = prof.columns.get(col)
        if col == target or c is None or column_role(str(col), c, target) in ("identifier", "pii", "empty"):
            continue
        vals = [_norm(v, nulls) for v in df[col]]
        pairs = [(v, y) for v, y in zip(vals, labels, strict=True) if y is not None]
        if len(pairs) < 20:
            continue
        baseline = Counter(y for _, y in pairs).most_common(1)[0][1] / len(pairs)
        # missingness
        missing_share = sum(1 for v, _ in pairs if v is None) / len(pairs)
        if 0.05 <= missing_share <= 0.95:
            acc, majority = _separation([(v is None, y) for v, y in pairs])
            if acc >= LEAKAGE_ACCURACY and acc > baseline and majority[True] != majority[False]:
                out.append({"column": str(col), "kind": "missingness", "accuracy": round(acc, 4),
                            "detail": f"empty -> {target} = '{majority[True]}', filled -> '{majority[False]}' "
                                      f"for {acc:.1%} of rows"})
                continue
        # value -> label
        present = [(v, y) for v, y in pairs if v is not None]
        distinct = {v for v, _ in present}
        if len(present) < 20 or not 2 <= len(distinct) <= min(50, 0.2 * len(present)):
            continue  # near-unique columns "predict" anything; leave those to the identifier check
        acc, _ = _separation(present)
        if acc >= LEAKAGE_ACCURACY and acc > baseline:
            out.append({"column": str(col), "kind": "value", "accuracy": round(acc, 4),
                        "detail": f"its {len(distinct)} values predict {target} for {acc:.1%} of rows"})
    return out


def ml_section(run: RunResult, data_run: RunResult | None = None) -> dict[str, Any]:
    """Target summary and leakage from `data_run` (the cleaned data, when exported); missing labels
    and label conflicts from `run`, so row numbers refer to the original file."""
    data_run = data_run or run
    nulls = normalise_nulls(run.rules.null_values)
    sheets = []
    for srules in run.rules.sheets:
        name = srules.sheet or run.wb.first_sheet.name
        sheet, prof = data_run.wb.sheets.get(name), data_run.profiles.get(name)
        if sheet is None or prof is None:
            continue
        entry: dict[str, Any] = {"sheet": name, "target": srules.target}
        if srules.target:
            entry["target_summary"] = target_summary(sheet, srules.target, nulls)
            entry["leakage_suspects"] = leakage_suspects(sheet, prof, srules.target, nulls)
            entry["missing_target_rows"] = [i.xl_row for i in run.validation.issues
                                            if i.sheet == name and i.rule_id == "target.missing"]
            entry["label_conflicts"] = sorted({tuple((i.context or {}).get("rows", [])) for i in run.validation.issues
                                               if i.sheet == name and i.rule_id == "target.conflict"})
        sheets.append(entry)
    return {"sheets": sheets}


# ----------------------------------------------------------------------------- typed data


def typed_frame(sheet: SheetData, schema: dict[str, Any], nulls) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """DataFrame with the schema dtypes. A column whose values cannot all be parsed stays text."""
    data: dict[str, Any] = {}
    kept_as_text = []
    for col in schema["columns"]:
        name, dtype = col["name"], col["dtype"]
        raw = list(sheet.df[name]) if name in sheet.df.columns else []
        texts = [to_text(v, nulls) for v in raw]
        converted: list[Any] | None = None
        bad = 0
        if dtype in ("Int64", "Float64"):
            nums = [to_number(v, nulls) for v in raw]
            bad = sum(1 for t, n in zip(texts, nums, strict=True) if t is not None and n is None)
            if dtype == "Int64" and not bad and any(n is not None and not float(n).is_integer() for n in nums):
                dtype = "Float64"
            converted = None if bad else [None if n is None else (int(n) if dtype == "Int64" else n) for n in nums]
        elif dtype == "datetime64[ns]":
            ds = [to_date(v, nulls) for v in raw]
            bad = sum(1 for t, d in zip(texts, ds, strict=True) if t is not None and d is None)
            converted = None if bad else ds
        elif dtype == "boolean":
            bs = [to_bool(v, nulls) for v in raw]
            bad = sum(1 for t, b in zip(texts, bs, strict=True) if t is not None and b is None)
            converted = None if bad else bs
        if bad:
            kept_as_text.append({"sheet": schema["sheet"], "column": name, "intended_dtype": col["dtype"],
                                 "unparseable_values": bad})
            data[name] = pd.array(texts, dtype="string")
        elif converted is not None and dtype == "datetime64[ns]":
            data[name] = pd.to_datetime(pd.Series(converted, dtype=object))
        elif converted is not None:
            data[name] = pd.array(converted, dtype=dtype)
        elif dtype == "category":
            data[name] = pd.Categorical(texts)
        else:
            data[name] = pd.array(texts, dtype="string")
    return pd.DataFrame(data), kept_as_text


def load_prepared(report_path: str | Path, sheet: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a cleaned CSV with the dtypes recorded in its cleaning report.

        df, schema = load_prepared("prepared/churn_cleaning_report.json")
        X = df[schema["features"]["numeric"] + schema["features"]["categorical"]]
        y = df[schema["excluded"]["target"]]
    """
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schemas = {s["sheet"]: s for s in report["columns"]}
    name = sheet or next(iter(schemas))
    schema = schemas[name]
    csv_name = report["dataset"]["files"][name]["csv"]
    path = report_path.parent / csv_name
    kept = {k["column"] for k in report["dataset"].get("kept_as_text", []) if k["sheet"] == name}
    dates = [c["name"] for c in schema["columns"] if c["dtype"] == "datetime64[ns]" and c["name"] not in kept]
    dtypes = {c["name"]: ("string" if c["name"] in kept or c["name"] in dates else c["dtype"])
              for c in schema["columns"]}
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
    for col in dates:
        df[col] = pd.to_datetime(df[col], errors="raise")
    return df, schema
