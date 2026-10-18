"""Cell-value helpers: null detection, kind classification and safe coercion.

Values arrive raw from openpyxl (str, int, float, bool, datetime, None), so
nothing is coerced at load time. These helpers decide what a value *is* and what
it *could be converted to*, which the profiler, the rule engine and the
correction suggesters all share.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

DEFAULT_NULLS = ("", "na", "n/a", "-", "null", "none", "nan", "#n/a")

_NUMERIC_TEXT = re.compile(r"^[+-]?[₹$€£]?\s?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$|^[+-]?\.\d+$")
_LEADING_ZERO_ID = re.compile(r"^0\d+$")
_BOOL_TEXT = {"true": True, "false": False, "yes": True, "no": False}

# Excel serial day 1 is 1900-01-01; with the 1900 leap-year bug, day 0 is 1899-12-30.
_EXCEL_EPOCH = datetime(1899, 12, 30)
SERIAL_MIN, SERIAL_MAX = 20000, 80000  # roughly 1954 .. 2119

_DATE_FORMATS_UNAMBIGUOUS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%Y",
    "%d-%b-%y",
)
_DAYFIRST = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
_ISO_START = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_NUMERIC_DMY = re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}$")
_HAS_MONTH_NAME = re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b")
_MONTHFIRST = ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y")


def is_null(v: Any, nulls: tuple[str, ...] | frozenset[str] = DEFAULT_NULLS) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str):
        return v.strip().lower() in nulls
    return False


def normalise_nulls(values: list[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset(DEFAULT_NULLS)
    return frozenset(s.strip().lower() for s in values)


def parse_number_text(s: str) -> float | None:
    t = s.strip()
    if not _NUMERIC_TEXT.match(t):
        return None
    t = t.replace(",", "").replace(" ", "")
    for sym in "₹$€£":
        t = t.replace(sym, "")
    try:
        return float(t)
    except ValueError:
        return None


def parse_date_text(s: str, dayfirst: bool = True) -> tuple[date | None, bool]:
    """Parse a text date. Returns (date, ambiguous).

    Ambiguous means the day-first and month-first readings are both valid and
    differ (03/04/2024): the result follows `dayfirst` but is flagged.
    """
    return _parse_date_cached(s.strip(), dayfirst)


@lru_cache(maxsize=65536)
def _parse_date_cached(t: str, dayfirst: bool) -> tuple[date | None, bool]:
    # Cheap shape checks first: strptime is slow, and most text is not a date.
    if not 6 <= len(t) <= 40 or not any(ch.isdigit() for ch in t):
        return None, False
    if _ISO_START.match(t):
        try:
            return datetime.fromisoformat(t.replace("/", "-")).date(), False  # handles time + timezone
        except ValueError:
            pass
        return _try_formats(t, _DATE_FORMATS_UNAMBIGUOUS), False
    if _NUMERIC_DMY.match(t):
        d_first = _try_formats(t, _DAYFIRST)
        m_first = _try_formats(t, _MONTHFIRST)
        if d_first and m_first:
            if d_first == m_first:
                return d_first, False
            return (d_first if dayfirst else m_first), True
        return d_first or m_first, False
    if _HAS_MONTH_NAME.search(t):
        return _try_formats(t, _DATE_FORMATS_UNAMBIGUOUS), False
    return None, False


def _try_formats(t: str, fmts: tuple[str, ...]) -> date | None:
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


_EMAIL = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
_PHONE_CHARS = re.compile(r"^\+?[\d\s\-().]{7,20}$")


def is_email(s: str) -> bool:
    return bool(_EMAIL.match(s.strip()))


def looks_like_phone(s: str) -> bool:
    t = s.strip()
    digits = sum(ch.isdigit() for ch in t)
    return bool(_PHONE_CHARS.match(t)) and 7 <= digits <= 15


def normalize_phone(s: str, digits: int = 10) -> str | None:
    """Digits-only national number, dropping separators and a leading country code or 0.

    Returns None when the value cannot be read as a phone number of `digits` digits.
    """
    t = s.strip()
    if not _PHONE_CHARS.match(t):
        return None
    d = re.sub(r"\D", "", t)
    if len(d) == digits:
        return d
    has_country_code = t.startswith("+") or t.startswith("00")
    if has_country_code and digits < len(d) <= digits + 4:
        return d[-digits:]  # +91 98765 43210, 0091-9876543210
    if t.startswith("0") and len(d) == digits + 1:
        return d[1:]  # national trunk prefix: 09876543210
    return None


def excel_serial_to_date(n: float) -> date:
    return (_EXCEL_EPOCH + timedelta(days=float(n))).date()


def is_serial_candidate(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    return SERIAL_MIN <= v <= SERIAL_MAX and float(v).is_integer()


def value_kind(v: Any, nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS) -> str:
    """Classify a raw cell value.

    Returns one of: null, bool, int, float, date, numeric_text, date_text,
    bool_text, text.
    """
    if is_null(v, nulls):
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "int" if v.is_integer() else "float"
    if isinstance(v, (datetime, date)):
        return "date"
    if isinstance(v, str):
        t = v.strip()
        if _LEADING_ZERO_ID.match(t):
            return "text"  # identifiers such as 00123 must stay text
        if parse_number_text(t) is not None:
            return "numeric_text"
        if t.lower() in _BOOL_TEXT:
            return "bool_text"
        if parse_date_text(t)[0] is not None:
            return "date_text"
        return "text"
    return "text"


def to_number(v: Any, nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS) -> float | None:
    if is_null(v, nulls) or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if _LEADING_ZERO_ID.match(v.strip()):
            return float(v.strip())
        return parse_number_text(v)
    return None


def to_date(
    v: Any,
    nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS,
    dayfirst: bool = True,
    allow_serial: bool = True,
) -> date | None:
    if is_null(v, nulls):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return parse_date_text(v, dayfirst)[0]
    if allow_serial and is_serial_candidate(v):
        return excel_serial_to_date(v)
    return None


def to_bool(v: Any, nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS) -> bool | None:
    if is_null(v, nulls):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return _BOOL_TEXT.get(v.strip().lower())
    return None


def to_text(v: Any, nulls: frozenset[str] | tuple[str, ...] = DEFAULT_NULLS) -> str | None:
    """Canonical text form used for comparisons, uniqueness and regex checks."""
    if is_null(v, nulls):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, datetime):
        return v.date().isoformat() if v.time() == datetime.min.time() else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v).strip()


def shape_pattern(s: str, max_len: int = 24) -> str:
    """Character-class shape of a string: 'ORD-00012' -> 'AAA-99999'."""
    out = []
    for ch in s[:max_len]:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A")
        elif ch.isspace():
            out.append("_")
        else:
            out.append(ch)
    if len(s) > max_len:
        out.append("…")
    return "".join(out)
