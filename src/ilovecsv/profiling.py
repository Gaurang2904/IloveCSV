"""Column and sheet profiling: types, missing values, distinct values, distributions."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .models import jsonable
from .reader import SheetData, WorkbookData, text_values
from .values import (
    DEFAULT_NULLS,
    is_email,
    is_null,
    is_serial_candidate,
    looks_like_phone,
    parse_number_text,
    shape_pattern,
    to_date,
    to_number,
    to_text,
    value_kind,
)

_ID_NAME = re.compile(r"(^|[_\s-])(id|code|no|num|number|ref|key)$", re.IGNORECASE)
_CAMEL_ID = re.compile(r"[a-z]ID$")
SEMANTIC_SAMPLE = 2000
_PLAIN_NUMBER = re.compile(r"^-?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")


@dataclass
class ColumnProfile:
    name: str
    letter: str
    inferred_type: str  # empty|int|float|date|bool|category|text|mixed
    type_share: float
    n_rows: int
    n_missing: int
    missing_rate: float
    n_distinct: int
    top_values: list[dict[str, Any]]
    kinds: dict[str, int]
    numeric: dict[str, Any] | None = None
    dates: dict[str, Any] | None = None
    lengths: dict[str, Any] | None = None
    patterns: list[dict[str, Any]] = field(default_factory=list)
    n_whitespace: int = 0
    case_variants: list[dict[str, Any]] = field(default_factory=list)
    n_numeric_text: int = 0
    n_date_text: int = 0
    n_serial_dates: int = 0
    is_identifier: bool = False
    # A value far more frequent than any typical value in a high-cardinality column,
    # e.g. a placeholder date such as 2020-01-01 00:00:00 used for unknown dates.
    dominant_value: dict[str, Any] | None = None
    semantic: str | None = None  # "email" | "phone" when the values look like contact details
    phone_digits: int | None = None  # most common digit count of local phone numbers
    sequential_id: bool = False  # surrogate row counter such as C001, C002, ... (ignored for duplicates)
    categories: list[dict[str, Any]] | None = None  # every distinct value of a category column

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SheetProfile:
    name: str
    n_rows: int
    n_cols: int
    header_row: int
    duplicate_rows: list[int]  # Excel rows that repeat an earlier row exactly
    dropped_blank_rows: list[int]
    dropped_blank_cols: list[str]
    columns: dict[str, ColumnProfile]

    @property
    def n_duplicate_rows(self) -> int:
        return len(self.duplicate_rows)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_duplicate_rows"] = self.n_duplicate_rows
        d["columns"] = [c.to_dict() for c in self.columns.values()]
        return d


def profile_workbook(
    wb: WorkbookData, nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS
) -> dict[str, SheetProfile]:
    profiles = {name: profile_sheet(s, nulls) for name, s in wb.sheets.items()}
    # Every CSV value is text (JSON dates too): only count numbers/dates written in a non-standard way.
    for name, prof in profiles.items():
        tv = text_values(wb, name)
        if tv is None:
            continue
        df = wb.sheets[name].df
        for col, c in prof.columns.items():
            vals = [v for v in df[col] if isinstance(v, str)]
            if tv == "all":
                c.n_numeric_text = sum(1 for v in vals if value_kind(v, nulls) == "numeric_text"
                                       and not _PLAIN_NUMBER.match(v.strip()))
            c.n_date_text = sum(1 for v in vals if value_kind(v, nulls) == "date_text"
                                and not _ISO_DATE.match(v.strip()))
    return profiles


def profile_sheet(
    sheet: SheetData, nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS
) -> SheetProfile:
    df = sheet.df
    cols = {c: profile_column(c, sheet.col_letters[c], df[c].tolist(), nulls) for c in df.columns}
    return SheetProfile(
        name=sheet.name,
        n_rows=len(df),
        n_cols=len(df.columns),
        header_row=sheet.header_row,
        duplicate_rows=duplicate_rows(sheet, nulls),
        dropped_blank_rows=list(sheet.dropped_blank_rows),
        dropped_blank_cols=list(sheet.dropped_blank_cols),
        columns=cols,
    )


def duplicate_rows(sheet: SheetData, nulls=DEFAULT_NULLS) -> list[int]:
    seen: set[tuple] = set()
    dups: list[int] = []
    for xl_row, row in zip(sheet.df.index, sheet.df.itertuples(index=False), strict=True):
        key = tuple(to_text(v, nulls) for v in row)
        if all(k is None for k in key):
            continue
        if key in seen:
            dups.append(int(xl_row))
        else:
            seen.add(key)
    return dups


def infer_type(kinds: Counter, values: list[Any], n_distinct: int) -> tuple[str, float]:
    n = sum(v for k, v in kinds.items() if k != "null")
    if n == 0:
        return "empty", 1.0
    dates = kinds["date"] + kinds["date_text"]
    if dates / n >= 0.5:
        dates += sum(1 for v in values if is_serial_candidate(v))
    numeric = kinds["int"] + kinds["float"] + kinds["numeric_text"]
    bools = kinds["bool"] + kinds["bool_text"]
    if dates / n >= 0.8:
        return "date", dates / n
    if numeric / n >= 0.8:
        integral = kinds["float"] == 0 and all(
            (parse_number_text(v) or 0).is_integer()
            for v in values
            if isinstance(v, str) and parse_number_text(v) is not None
        )
        return ("int" if integral else "float"), numeric / n
    if bools / n >= 0.8:
        return "bool", bools / n
    text = kinds["text"]
    if text / n >= 0.8:
        if n_distinct <= 20 and n_distinct <= 0.5 * n:
            return "category", text / n
        return "text", text / n
    best = max(("date", dates), ("number", numeric), ("text", text), key=lambda t: t[1])
    return "mixed", best[1] / n


def _kinds(values: list[Any], nulls) -> Counter:
    """value_kind for every value, with a fast path for the plain numbers of binary formats."""
    kinds: Counter = Counter()
    numeric_codes = getattr(nulls, "numeric", None)
    for v in values:
        t = type(v)
        if t is float:
            if v != v or (numeric_codes and v in numeric_codes):
                kinds["null"] += 1
            else:
                kinds["int" if v.is_integer() else "float"] += 1
        elif t is int:
            kinds["null" if numeric_codes and v in numeric_codes else "int"] += 1
        elif v is None:
            kinds["null"] += 1
        else:
            kinds[value_kind(v, nulls)] += 1
    return kinds


def profile_column(name: str, letter: str, values: list[Any], nulls=DEFAULT_NULLS) -> ColumnProfile:
    kinds = _kinds(values, nulls)
    n_rows = len(values)
    n_missing = kinds["null"]
    texts = [to_text(v, nulls) for v in values]
    non_null_texts = [t for t in texts if t is not None]
    counts = Counter(non_null_texts)
    n_distinct = len(counts)
    inferred, share = infer_type(kinds, values, n_distinct)

    prof = ColumnProfile(
        name=name,
        letter=letter,
        inferred_type=inferred,
        type_share=round(share, 4),
        n_rows=n_rows,
        n_missing=n_missing,
        missing_rate=round(n_missing / n_rows, 4) if n_rows else 0.0,
        n_distinct=n_distinct,
        top_values=[{"value": v, "count": c} for v, c in counts.most_common(10)],
        kinds={k: v for k, v in kinds.items()},
        n_numeric_text=kinds["numeric_text"],
        n_date_text=kinds["date_text"],
    )

    if inferred in ("int", "float"):
        nums = np.array([x for x in (to_number(v, nulls) for v in values) if x is not None])
        if nums.size:
            # Bin the 1st-99th percentile so one outlier does not squash the whole chart.
            lo, hi = np.quantile(nums, [0.01, 0.99]) if nums.size >= 50 else (nums.min(), nums.max())
            inside = nums[(nums >= lo) & (nums <= hi)]
            counts_h, edges = np.histogram(inside, bins=min(10, max(1, len(np.unique(inside)))))
            q = np.quantile(nums, [0.05, 0.25, 0.5, 0.75, 0.95])
            # "Bulk" range: values within robust z <= 3.5, so outliers do not define what is normal.
            med = float(np.median(nums))
            mad = float(np.median(np.abs(nums - med)))
            bulk = nums[np.abs(0.6745 * (nums - med) / mad) <= 3.5] if mad > 0 else nums
            b_lo, b_hi = float(bulk.min()), float(bulk.max())
            width = (b_hi - b_lo) or abs(b_hi) or 1.0
            prof.numeric = {
                "bulk_min": b_lo,
                "bulk_max": b_hi,
                "count": int(nums.size),
                "negative_share": round(float((nums < 0).mean()), 4),
                "n_negative": int((nums < 0).sum()),
                # values outside the bulk widened by its own width (what a suggested bound would flag)
                "n_below_window": int((nums < b_lo - width).sum()),
                "n_above_window": int((nums > b_hi + width).sum()),
                "min": float(nums.min()),
                "max": float(nums.max()),
                "mean": float(nums.mean()),
                "std": float(nums.std(ddof=1)) if nums.size > 1 else 0.0,
                "p05": float(q[0]),
                "p25": float(q[1]),
                "median": float(q[2]),
                "p75": float(q[3]),
                "p95": float(q[4]),
                "histogram": {
                    "edges": [float(e) for e in edges],
                    "counts": [int(c) for c in counts_h],
                    "outside": int(nums.size - inside.size),
                },
            }
    if inferred == "date":
        ds = [d for d in (to_date(v, nulls) for v in values) if d is not None]
        prof.n_serial_dates = sum(1 for v in values if is_serial_candidate(v))
        if ds:
            prof.dates = {"min": min(ds).isoformat(), "max": max(ds).isoformat()}

    raw_strings = [v for v in values if isinstance(v, str) and not is_null(v, nulls)]
    if raw_strings:
        lengths = [len(s) for s in raw_strings]
        mode, mode_n = Counter(lengths).most_common(1)[0]
        prof.lengths = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "mode": mode,
            "mode_share": round(mode_n / len(lengths), 4),
            "off_mode": len(lengths) - mode_n,
        }
        prof.n_whitespace = sum(1 for s in raw_strings if has_whitespace_problem(s))
        prof.patterns = [
            {"pattern": p, "count": c}
            for p, c in Counter(shape_pattern(s.strip()) for s in raw_strings).most_common(5)
        ]
        prof.case_variants = case_variant_groups(raw_strings)

    if inferred == "category":  # at most 20 distinct values by definition: keep them all
        prof.categories = [{"value": v, "count": c} for v, c in counts.most_common()]
    prof.semantic = None if inferred == "date" else detect_semantic(name, non_null_texts)
    # A surrogate row counter is unique per row; tenure 1..72 repeated across rows is a quantity.
    prof.sequential_id = (inferred in ("text", "int") and n_distinct >= 0.95 * len(non_null_texts)
                          and is_sequential_id(non_null_texts))
    if prof.semantic == "phone":
        local = [sum(ch.isdigit() for ch in t) for t in non_null_texts if not t.startswith(("+", "00"))]
        prof.phone_digits = Counter(local).most_common(1)[0][0] if local else None
    prof.is_identifier = _looks_like_identifier(name, prof)
    if inferred in ("date", "int", "float"):  # where placeholders like 1900-01-01 or 9999 hide
        prof.dominant_value = _dominant_value(counts, len(non_null_texts))
    prof.top_values = [{"value": jsonable(t["value"]), "count": t["count"]} for t in prof.top_values]
    return prof


def has_whitespace_problem(s: str) -> bool:
    return s != s.strip() or "  " in s


def collapse_ws(s: str) -> str:
    return " ".join(s.split())


def text_style(s: str) -> str:
    if sum(ch.isalpha() for ch in s) < 2:
        return "other"
    if s.isupper():
        return "upper"
    if s.islower():
        return "lower"
    return "title" if s.istitle() else "mixed"


def column_style(spellings: Counter) -> str | None:
    """The capitalisation most values in the column use (weighted by rows), if one dominates."""
    weights: Counter = Counter()
    for s, n in spellings.items():
        style = text_style(s)
        if style != "other":
            weights[style] += n
    if not weights:
        return None
    style, n = weights.most_common(1)[0]
    return style if n / sum(weights.values()) >= 0.5 else None


def pick_spelling(variants: Counter, style: str | None = None) -> str:
    """Canonical spelling for one group of case/spacing variants.

    Prefer variants written in the column's dominant style (so 'Yes' next to 'No', even when 'YES'
    is more frequent), then the most frequent, then Title Case. 'USA' stays 'USA' when no variant in
    the dominant style exists.
    """
    pool = {s: n for s, n in variants.items() if style and text_style(s) == style} or dict(variants)
    return max(pool.items(), key=lambda kv: (kv[1], kv[0].istitle(), not (kv[0].isupper() or kv[0].islower())))[0]


def canonical_spellings(strings: list[str]) -> dict[str, str]:
    """casefolded value -> canonical spelling, for every value of one column."""
    spelled = Counter(collapse_ws(s) for s in strings if isinstance(s, str) and s.strip())
    style = column_style(spelled)
    groups: dict[str, Counter] = defaultdict(Counter)
    for s, n in spelled.items():
        groups[s.casefold()][s] += n
    return {k: pick_spelling(c, style) for k, c in groups.items()}


def case_variant_groups(strings: list[str], limit: int = 10) -> list[dict[str, Any]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for s in strings:
        groups[collapse_ws(s).casefold()][s] += 1
    canonical = canonical_spellings(strings)
    out = []
    for key, counter in groups.items():
        if len(counter) > 1:
            out.append(
                {
                    "canonical": canonical[key],
                    "variants": [{"value": v, "count": c} for v, c in counter.most_common()],
                }
            )
    out.sort(key=lambda g: -sum(v["count"] for v in g["variants"]))
    return out[:limit]


def _looks_like_identifier(name: str, prof: ColumnProfile) -> bool:
    non_null = prof.n_rows - prof.n_missing
    if non_null == 0:
        return False
    unique_ratio = prof.n_distinct / non_null
    n = name.strip()
    if (_ID_NAME.search(n) or _CAMEL_ID.search(n)) and unique_ratio > 0.5:
        return True
    same_length = prof.lengths is not None and prof.lengths["mode_share"] >= 0.99
    if prof.inferred_type == "text" and unique_ratio == 1.0 and non_null >= 20 and same_length:
        return True  # hashes, codes: every value distinct and (almost) all the same length
    return prof.inferred_type == "text" and unique_ratio > 0.98 and len(prof.patterns) <= 2


def id_like_name(name: str) -> bool:
    n = name.strip()
    return bool(_ID_NAME.search(n) or _CAMEL_ID.search(n))


_PREFIXED_NUMBER = re.compile(r"^(\D*?)(\d+)$")


def is_sequential_id(texts: list[str]) -> bool:
    """A surrogate row counter: one prefix + a number, numbers mostly consecutive (C001, C002, ...)."""
    if len(texts) < 5:
        return False
    parts = [m.groups() for t in texts if (m := _PREFIXED_NUMBER.match(t))]
    if len(parts) < 0.95 * len(texts):
        return False
    prefix, count = Counter(p for p, _ in parts).most_common(1)[0]
    if count < 0.95 * len(parts):
        return False
    nums = sorted({int(n) for p, n in parts if p == prefix})
    if len(nums) < 5:
        return False
    steps = [b - a for a, b in zip(nums, nums[1:], strict=False)]
    return sum(1 for s in steps if s == 1) >= 0.9 * len(steps)


def detect_semantic(name: str, texts: list[str]) -> str | None:
    """Recognise email and phone columns from their values (and, more leniently, their name).

    Decided on an evenly spread sample: the result is a share, and scanning a million measurement
    values for '@' or phone shapes would cost seconds for nothing."""
    if not texts:
        return None
    if len(texts) > SEMANTIC_SAMPLE:
        step = len(texts) / SEMANTIC_SAMPLE
        texts = [texts[int(i * step)] for i in range(SEMANTIC_SAMPLE)]
    lname = name.lower()
    if any("@" in t for t in texts):
        email_share = sum(1 for t in texts if is_email(t)) / len(texts)
        if email_share >= 0.8 or ("mail" in lname and email_share >= 0.5):
            return "email"
    if sum(1 for t in texts if "." in t and t.replace(".", "", 1).lstrip("-").isdigit()) > 0.5 * len(texts):
        return None  # decimal numbers are measurements, not phone numbers
    phone_share = sum(1 for t in texts if looks_like_phone(t)) / len(texts)
    named = any(k in lname for k in ("phone", "mobile", "tel", "contact no", "cell"))
    if (named and phone_share >= 0.6) or (phone_share >= 0.9 and any(not t.isdigit() for t in texts)):
        return "phone"
    return None


def _dominant_value(counts: Counter, n: int) -> dict[str, Any] | None:
    if len(counts) < 50 or n == 0:
        return None
    (value, top), *_ = counts.most_common(1)
    typical = float(np.median(list(counts.values())))
    if top / n >= 0.01 and top >= 20 * max(typical, 1.0):
        return {"value": jsonable(value), "count": top, "share": round(top / n, 4),
                "typical_count": typical}
    return None
