"""Dataset documentation reader: README, data dictionary, dataset card, codebook.

Research datasets explain themselves in side files: what V, T or RUL mean, units, how missing
values are coded (-999, "NA"). This module finds those files and shows them for review. Nothing
here is applied automatically: a sentence like "missing values are coded as -999" becomes a
*suggestion* to add -999 to the recipe's null_values, which the user decides.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import PurePosixPath
from typing import Any

DOC_NAME = re.compile(
    r"(?i)(readme|read_me|license|licence|changelog|notes|description|describe|codebook|code_book|"
    r"data[_ -]?dictionary|dictionary|schema|metadata|data[_ -]?card|dataset[_ -]?card|dataset_info|"
    r"columns|variables|attributes|about|citation)")
TEXT_EXT = {".md", ".txt", ".rst", ".text", ".yaml", ".yml", ".json", ".html", ".htm", ".names", ".info"}
TABLE_EXT = {".csv", ".tsv"}
MAX_TEXT = 20_000

_MISSING_PATTERNS = [
    re.compile(r"missing(?:\s+values?)?\s+(?:are|is|were|was)?\s*(?:coded|encoded|represented|marked|denoted|"
               r"recorded|given|shown|indicated)\s+(?:as|by|with)\s+[\"'`]?([^\s\"'`,;)]+)", re.I),
    re.compile(r"[\"'`]?([-−]?\d{2,}(?:\.\d+)?|NA|N/A|NaN|null|NULL|None|\?)[\"'`]?\s+(?:indicates|means|denotes|"
               r"represents|signifies|is used for|for)\s+(?:a\s+)?(?:missing|unknown|not available)", re.I),
]
_ALLOWED_TOKEN = re.compile(r"^(?:[-−]?\d+(?:\.\d+)?|[A-Za-z/?#.]{1,12})$")


def is_doc_file(name: str) -> bool:
    p = PurePosixPath(name)
    ext = p.suffix.lower()
    if ext in TEXT_EXT and (DOC_NAME.search(p.stem) or ext in {".md", ".rst", ".names", ".info"}):
        return True
    return ext in TABLE_EXT and bool(re.search(r"(?i)(dictionary|codebook|schema|variables|columns|metadata)", p.stem))


def doc_kind(name: str) -> str:
    stem = PurePosixPath(name).stem.lower()
    if re.search(r"dictionary|codebook|code_book|schema|variables|columns|attributes", stem):
        return "data_dictionary"
    if re.search(r"card|dataset_info|metadata", stem):
        return "dataset_card"
    if re.search(r"licen[cs]e", stem):
        return "license"
    if "readme" in stem or "read_me" in stem:
        return "readme"
    return "documentation"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")  # pragma: no cover


def missing_value_hints(text: str) -> list[dict[str, str]]:
    hints: dict[str, str] = {}
    for pattern in _MISSING_PATTERNS:
        for m in pattern.finditer(text):
            token = m.group(1).replace("−", "-").strip(".")
            if token and _ALLOWED_TOKEN.match(token) and token.lower() not in hints:
                start, end = max(0, m.start() - 60), min(len(text), m.end() + 60)
                hints[token.lower()] = (token, " ".join(text[start:end].split()))
    return [{"token": tok, "context": ctx} for tok, ctx in hints.values()]


def _dictionary_rows(text: str) -> list[dict[str, str]]:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    return [{str(k).strip(): (v or "").strip() for k, v in r.items() if k} for r in rows[:300]]


def read_doc(name: str, data: bytes, dataset_columns: list[str] | None = None) -> dict[str, Any]:
    ext = PurePosixPath(name).suffix.lower()
    text = _decode(data[: MAX_TEXT * 4])
    entry: dict[str, Any] = {"path": name, "kind": doc_kind(name), "size_bytes": len(data)}
    if ext in TABLE_EXT:
        rows = _dictionary_rows(text)
        entry["rows"] = rows
        text = "\n".join(" | ".join(r.values()) for r in rows)
    entry["text"] = text[:MAX_TEXT] + ("\n… (truncated)" if len(text) > MAX_TEXT else "")
    entry["missing_value_hints"] = missing_value_hints(text)
    if dataset_columns:
        low = text.lower()
        entry["columns_mentioned"] = [c for c in dataset_columns
                                      if len(c) > 1 and re.search(rf"(?<![\w]){re.escape(c.lower())}(?![\w])", low)]
    return entry
