"""Rule engine: evaluate a RuleSet against a workbook and emit Issues.

Every issue produced here is a *confirmed* problem (kind="error"): a stated rule
was broken. Statistical suspicion lives in ``anomalies.py`` and is kept separate.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from ..models import ERROR, Issue
from ..profiling import (
    SheetProfile,
    collapse_ws,
    has_whitespace_problem,
    id_like_name,
    profile_workbook,
)
from ..reader import SheetData, WorkbookData
from ..values import (
    is_email,
    is_null,
    is_serial_candidate,
    normalise_nulls,
    normalize_phone,
    parse_date_text,
    to_bool,
    to_date,
    to_number,
    to_text,
    value_kind,
)
from .expr import CompiledExpr, ExprError, compile_expr
from .schema import ColumnRule, RuleSet, SheetRules, TableRule

NUMERIC_TYPES = {"int", "float", "number"}
# CSV has no cell types, so plain numbers and ISO dates are already canonical there.
_CANONICAL_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_CANONICAL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")


@dataclass
class ValidationResult:
    issues: list[Issue]
    rule_errors: list[dict[str, Any]] = field(default_factory=list)
    checked: dict[str, dict[str, int]] = field(default_factory=dict)
    profiles: dict[str, SheetProfile] = field(default_factory=dict)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.kind == ERROR]


def validate(
    wb: WorkbookData, rules: RuleSet, profiles: dict[str, SheetProfile] | None = None
) -> ValidationResult:
    nulls = normalise_nulls(rules.null_values)
    profiles = profiles or profile_workbook(wb, nulls)
    result = ValidationResult(issues=[], profiles=profiles)
    seen: set[str] = set()
    for srules in rules.sheets:
        name = srules.sheet or wb.first_sheet.name
        sheet = wb.sheets.get(name)
        if sheet is None:
            _add(result, seen, Issue(ERROR, name, "sheet_missing", "error",
                                     f"Expected sheet '{name}' was not found in the workbook."))
            continue
        v = SheetValidator(wb, sheet, srules, rules, profiles[name], nulls)
        for issue in v.run():
            _add(result, seen, issue)
        result.rule_errors.extend(v.rule_errors)
        result.checked[name] = {"rows": len(sheet.df), "columns": len(sheet.df.columns)}
    return result


def _add(result: ValidationResult, seen: set[str], issue: Issue) -> None:
    if issue.id not in seen:
        seen.add(issue.id)
        result.issues.append(issue)


class TypedColumns:
    """Typed views of raw columns for expression evaluation (cached)."""

    def __init__(self, sheet: SheetData, srules: SheetRules, profile: SheetProfile, nulls, dayfirst: bool):
        self.sheet, self.srules, self.profile = sheet, srules, profile
        self.nulls, self.dayfirst = nulls, dayfirst
        self._cache: dict[str, pd.Series] = {}

    def column_type(self, col: str) -> str:
        declared = self.srules.columns.get(col)
        if declared and declared.type:
            return declared.type
        prof = self.profile.columns.get(col)
        return prof.inferred_type if prof else "text"

    def __call__(self, col: str) -> pd.Series:
        if col in self._cache:
            return self._cache[col]
        if col not in self.sheet.df.columns:
            raise KeyError(col)
        raw = self.sheet.df[col]
        t = self.column_type(col)
        if t in NUMERIC_TYPES:
            s = pd.Series([to_number(v, self.nulls) for v in raw], index=raw.index, dtype="float64")
        elif t == "date":
            ds = [to_date(v, self.nulls, self.dayfirst) for v in raw]
            s = pd.to_datetime(pd.Series(ds, index=raw.index, dtype=object))
        elif t == "bool":
            s = pd.Series([to_bool(v, self.nulls) for v in raw], index=raw.index, dtype="boolean")
        else:
            s = pd.Series([to_text(v, self.nulls) for v in raw], index=raw.index, dtype=object)
        self._cache[col] = s
        return s

    def evaluate(self, expr: CompiledExpr) -> pd.Series:
        return expr.evaluate(self, self.sheet.df.index)


class SheetValidator:
    def __init__(self, wb: WorkbookData, sheet: SheetData, srules: SheetRules, rules: RuleSet,
                 profile: SheetProfile, nulls):
        self.wb, self.sheet, self.srules, self.rules = wb, sheet, srules, rules
        self.profile, self.nulls = profile, nulls
        self.typed = TypedColumns(sheet, srules, profile, nulls, rules.dayfirst)
        self.issues: list[Issue] = []
        self.rule_errors: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ emit helpers
    def emit(self, rule_id: str, severity: str, message: str, xl_row: int | None = None,
             column: str | None = None, value: Any = None, context: dict[str, Any] | None = None) -> None:
        if xl_row is not None and column is not None:
            if self.sheet.is_formula(xl_row, column) and is_null(value, self.nulls):
                return  # formula without a cached value (file never recalculated)
            cell_ref = self.sheet.cell_ref(xl_row, column)
        elif xl_row is not None:
            cell_ref = self.sheet.row_ref(xl_row)
        else:
            cell_ref = None
        self.issues.append(Issue(ERROR, self.sheet.name, rule_id, severity, message,
                                 xl_row=xl_row, column=column, cell_ref=cell_ref, value=value, context=context))

    def rule_error(self, rule_id: str, error: str) -> None:
        self.rule_errors.append({"sheet": self.sheet.name, "rule_id": rule_id, "error": error})

    # ------------------------------------------------------------------ run
    def run(self) -> list[Issue]:
        df = self.sheet.df
        for col, rule in self.srules.columns.items():
            if col not in df.columns:
                self.emit("column_missing", "error", f"Expected column '{col}' is missing.", column=col)
                continue
            self._column(col, rule)
        if self.srules.key:
            self._unique_together("key.duplicate", self.srules.key, "error", None)
        if self.srules.checks.whitespace:
            self._whitespace()
        if self.srules.checks.duplicates:
            self._duplicate_rows()
        if self.srules.checks.case_variants:
            self._case_variants()
        if self.srules.checks.null_tokens:
            self._null_tokens()
        if self.srules.target:
            self._target_checks(self.srules.target)
        for trule in self.srules.rules:
            try:
                self._table_rule(trule)
            except ExprError as e:
                self.rule_error(trule.id, str(e))
        return self.issues

    # ------------------------------------------------------------------ column rules
    def _column(self, col: str, rule: ColumnRule) -> None:
        items = list(self.sheet.df[col].items())
        sev = rule.severity
        msg = rule.message

        present: list[tuple[int, Any]] = []
        for xl_row, v in items:
            if is_null(v, self.nulls):
                if rule.required:
                    self.emit("required", sev, msg or f"'{col}' is required but empty.", xl_row, col, v)
            else:
                present.append((int(xl_row), v))

        if rule.type:
            self._type(col, rule, present)

        bounds_are_dates = isinstance(rule.min, date) or isinstance(rule.max, date)
        as_date = rule.type == "date" or bounds_are_dates or rule.not_future
        if rule.min is not None or rule.max is not None:
            lo = _as_date(rule.min) if as_date else rule.min
            hi = _as_date(rule.max) if as_date else rule.max
            for xl_row, v in present:
                x = to_date(v, self.nulls, self.rules.dayfirst) if as_date else to_number(v, self.nulls)
                if x is None:
                    continue  # unparseable values are reported by the type rule
                if lo is not None and x < lo:
                    self.emit("range", sev, msg or f"{_fmt(x)} is below the minimum {_fmt(lo)}.", xl_row, col, v)
                elif hi is not None and x > hi:
                    self.emit("range", sev, msg or f"{_fmt(x)} is above the maximum {_fmt(hi)}.", xl_row, col, v)
        if rule.not_future:
            today = date.today()
            for xl_row, v in present:
                d = to_date(v, self.nulls, self.rules.dayfirst)
                if d is not None and d > today:
                    self.emit("not_future", sev, msg or f"{d.isoformat()} is in the future.", xl_row, col, v)

        aliases = {collapse_ws(k).casefold(): canon for k, canon in (rule.aliases or {}).items()}
        aliased: set[int] = set()
        if aliases:
            for xl_row, v in present:
                canon = aliases.get(collapse_ws(to_text(v, self.nulls)).casefold())
                if canon is not None:
                    aliased.add(xl_row)
                    self.emit("allowed.alias", "warning",
                              f"'{to_text(v, self.nulls)}' is a known variant of '{canon}'.", xl_row, col, v)

        if rule.format == "email":
            for xl_row, v in present:
                t = to_text(v, self.nulls)
                if not is_email(t):
                    self.emit("format.email", sev, msg or f"'{t}' is not a valid email address.", xl_row, col, v)
                elif t != t.lower():
                    self.emit("format.email_case", "warning",
                              f"Email '{t}' has capital letters; addresses are compared in lowercase.",
                              xl_row, col, v)
        elif rule.format == "phone":
            for xl_row, v in present:
                t = to_text(v, self.nulls)
                norm = normalize_phone(t, rule.phone_digits)
                numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
                if norm is None:
                    n = sum(ch.isdigit() for ch in t)
                    self.emit("format.phone", sev, msg or f"'{t}' is not a {rule.phone_digits}-digit phone number "
                              f"({n} digits).", xl_row, col, v)
                elif numeric:
                    self.emit("format.phone_numeric", "warning",
                              f"Phone {t} is stored as a number; store it as text so leading zeros and "
                              "formatting are kept.", xl_row, col, v)
                elif norm != t:
                    self.emit("format.phone_style", "warning",
                              f"Phone written as '{t}'; the standard form is {norm}.", xl_row, col, v)

        if rule.allowed is not None:
            allowed = set(rule.allowed)
            folded = {collapse_ws(a).casefold(): a for a in rule.allowed}
            for xl_row, v in present:
                t = to_text(v, self.nulls)
                if t in allowed or xl_row in aliased:
                    continue
                canonical = folded.get(collapse_ws(t).casefold())
                if canonical is not None:
                    if not rule.case_insensitive:
                        self.emit("allowed.noncanonical", "warning",
                                  f"'{t}' should be written '{canonical}'.", xl_row, col, v)
                    continue
                self.emit("allowed", sev, msg or f"'{t}' is not an allowed value for '{col}'.", xl_row, col, v)

        if rule.regex:
            pattern = re.compile(rule.regex)
            for xl_row, v in present:
                t = to_text(v, self.nulls)
                if not pattern.fullmatch(t):
                    self.emit("regex", sev, msg or f"'{t}' does not match the pattern {rule.regex}.", xl_row, col, v)

        if rule.min_length is not None or rule.max_length is not None:
            for xl_row, v in present:
                n = len(to_text(v, self.nulls))
                if (rule.min_length is not None and n < rule.min_length) or (
                    rule.max_length is not None and n > rule.max_length
                ):
                    self.emit("length", sev, msg or f"Length {n} is outside "
                              f"{rule.min_length or 0}..{rule.max_length or '∞'}.", xl_row, col, v)

        if rule.unique:
            first: dict[str, int] = {}
            for xl_row, v in present:
                t = to_text(v, self.nulls)
                if t in first:
                    self.emit("unique", sev, msg or f"Duplicate value '{t}' (first seen in row {first[t]}).",
                              xl_row, col, v)
                else:
                    first[t] = xl_row

    def _type(self, col: str, rule: ColumnRule, present: list[tuple[int, Any]]) -> None:
        t, sev, msg = rule.type, rule.severity, rule.message
        csv = self.wb.file_type == "csv"
        for xl_row, v in present:
            k = value_kind(v, self.nulls)
            if t == "text":
                continue
            if csv and isinstance(v, str):
                if k == "numeric_text" and _CANONICAL_NUMBER.match(v.strip()):
                    k = "float" if "." in v and not float(v).is_integer() else "int"
                elif k == "date_text" and _CANONICAL_DATE.match(v.strip()):
                    k = "date"
                elif k == "bool_text" and v.strip().upper() in ("TRUE", "FALSE"):
                    k = "bool"
            if t in NUMERIC_TYPES:
                if k == "int" or (k == "float" and t != "int"):
                    continue
                if k == "numeric_text":
                    num = to_number(v, self.nulls)
                    if t == "int" and not float(num).is_integer():
                        self.emit("type", sev, msg or f"'{v}' is not a whole number.", xl_row, col, v)
                    else:
                        self.emit("type.stored_as_text", "warning", f"Number stored as text: '{v}'.", xl_row, col, v)
                    continue
                if k == "float" and t == "int":
                    self.emit("type", sev, msg or f"{v} is not a whole number.", xl_row, col, v)
                    continue
                self.emit("type", sev, msg or f"'{v}' is not a number.", xl_row, col, v)
            elif t == "date":
                if k == "date":
                    continue
                if k == "date_text":
                    _, ambiguous = parse_date_text(v, self.rules.dayfirst)
                    note = " (ambiguous day/month order)" if ambiguous else ""
                    self.emit("type.date_as_text", "warning", f"Date stored as text: '{v}'{note}.", xl_row, col, v)
                    continue
                if is_serial_candidate(v):
                    self.emit("type.date_serial", "warning",
                              f"Excel serial number {v} where a date is expected.", xl_row, col, v)
                    continue
                self.emit("type", sev, msg or f"'{v}' is not a date.", xl_row, col, v)
            elif t == "bool":
                if k == "bool":
                    continue
                if k == "bool_text" or (isinstance(v, str) and to_bool(v, self.nulls) is not None):
                    # yes/no are accepted for a column *declared* bool, never inferred as one
                    self.emit("type.stored_as_text", "warning", f"Boolean stored as text: '{v}'.", xl_row, col, v)
                    continue
                self.emit("type", sev, msg or f"'{v}' is not true/false.", xl_row, col, v)

    # ------------------------------------------------------------------ sheet checks
    def _whitespace(self) -> None:
        for col in self.sheet.df.columns:
            for xl_row, v in self.sheet.df[col].items():
                if isinstance(v, str) and not is_null(v, self.nulls) and has_whitespace_problem(v):
                    self.emit("format.whitespace", "warning",
                              f"Extra spaces in '{v}'.", int(xl_row), col, v)

    def duplicate_ignored_columns(self) -> list[str]:
        cfg = self.srules.checks.duplicate_ignore
        if cfg is not None:
            return [c for c in cfg if c in self.sheet.df.columns]
        # Surrogate ids only: sequential counters such as C001, C002 or 1, 2, 3 that number the rows.
        # Natural keys (a SHA-256, an email, a name) are part of what makes two records the same.
        # A table has one surrogate key, so ignore at most one counter (preferring one named like an id).
        counters = [c for c, p in self.profile.columns.items() if p.sequential_id]
        counters.sort(key=lambda c: not id_like_name(str(c)))
        ignored = counters[:1]
        # Keep at least two columns to compare, or every short row would look like a duplicate.
        return ignored if len(self.sheet.df.columns) - len(ignored) >= 2 else []

    def duplicate_columns(self) -> tuple[list[str], list[str]]:
        """(compared columns, ignored columns) for the duplicate check."""
        df = self.sheet.df
        chosen = self.srules.checks.duplicate_columns
        if chosen:
            missing = [c for c in chosen if c not in df.columns]
            if missing:
                self.rule_error("row.duplicate", f"duplicate_columns not in sheet: {missing}")
            compared = [c for c in chosen if c in df.columns]
            return compared, [c for c in df.columns if c not in compared]
        ignored = self.duplicate_ignored_columns()
        return [c for c in df.columns if c not in ignored], ignored

    def _duplicate_rows(self) -> None:
        """Rows that repeat an earlier row on the compared columns (by default: all except a
        surrogate id, ignoring case and spacing). Each issue records both rows for a preview."""
        df = self.sheet.df
        cols, ignored = self.duplicate_columns()
        if not cols:
            return
        normalized = self.srules.checks.duplicate_match == "normalized"

        def norm(v: Any) -> str | None:
            t = to_text(v, self.nulls)
            return None if t is None else (collapse_ws(t).casefold() if normalized else t)

        parts = []
        if ignored:
            parts.append(f"ignoring {', '.join(map(str, ignored))}")
        if normalized:
            parts.append("case and spacing ignored")
        note = f" ({'; '.join(parts)})" if parts else ""
        first: dict[tuple, int] = {}
        raw_rows = {int(r): row for r, row in zip(df.index, df.itertuples(index=False), strict=True)}
        for xl_row, row in zip(df.index, df[cols].itertuples(index=False), strict=True):
            key = tuple(norm(v) for v in row)
            if all(k is None for k in key):
                continue
            if key in first:
                orig = first[key]
                exact = tuple(raw_rows[orig]) == tuple(raw_rows[int(xl_row)])  # every cell identical, ids too
                text = f"Row is an exact copy of row {orig}." if exact else f"Row repeats row {orig}{note}."
                self.emit("row.duplicate", "error", text, int(xl_row),
                          context={"duplicate_of": orig, "exact": exact, "compared_columns": [str(c) for c in cols],
                                   "ignored_columns": [str(c) for c in ignored],
                                   "match": self.srules.checks.duplicate_match})
            else:
                first[key] = int(xl_row)

    def _null_tokens(self) -> None:
        """Text such as 'N/A', '-' or 'null' standing in for a missing value.

        Validation already treats these as missing; the issue exists so the cleaned file holds a
        real empty cell, which pandas and other tools read as missing without extra settings.
        """
        for col in self.sheet.df.columns:
            for xl_row, v in self.sheet.df[col].items():
                if isinstance(v, str) and is_null(v, self.nulls) and not self.sheet.is_formula(int(xl_row), col):
                    shown = v if v.strip() else "(spaces only)"
                    self.emit("format.null_token", "warning",
                              f"Placeholder '{shown}' stands for a missing value.", int(xl_row), col, v)

    def _target_checks(self, target: str) -> None:
        df = self.sheet.df
        if target not in df.columns:
            self.emit("column_missing", "error", f"Target column '{target}' is missing.", column=target)
            return
        for xl_row, v in df[target].items():
            if is_null(v, self.nulls):
                self.emit("target.missing", "error",
                          "No target label: this row cannot be used for supervised training.",
                          int(xl_row), target, v)
        # Identical features, different labels: contradictory training examples.
        ignored = self.duplicate_ignored_columns()  # a surrogate id would make every row unique
        features = [c for c in df.columns if c != target and c not in ignored]
        if not features:
            return

        def norm(v: Any) -> str | None:
            t = to_text(v, self.nulls)
            return None if t is None else collapse_ws(t).casefold()

        groups: dict[tuple, list[int]] = defaultdict(list)
        for xl_row, row in zip(df.index, df[features].itertuples(index=False), strict=True):
            key = tuple(norm(v) for v in row)
            if any(k is not None for k in key):
                groups[key].append(int(xl_row))
        for rows in groups.values():
            labels = {norm(df.at[r, target]) for r in rows} - {None}
            if len(rows) < 2 or len(labels) < 2:
                continue
            shown = sorted({str(to_text(df.at[r, target], self.nulls)) for r in rows})
            for r in rows:
                others = [o for o in rows if o != r]
                self.emit("target.conflict", "warning",
                          f"Same features as row(s) {', '.join(map(str, others))} but a different label "
                          f"({' vs '.join(shown)}).", r, target, df.at[r, target],
                          context={"rows": rows, "labels": shown})

    def _case_variants(self) -> None:
        """Same value written with different capitalisation in one column (mumbai / MUMBAI / Mumbai)."""
        for col in self.sheet.df.columns:
            rule = self.srules.columns.get(col)
            prof = self.profile.columns.get(col)
            if prof is None or prof.is_identifier or prof.inferred_type not in ("text", "category", "mixed"):
                continue
            if rule is not None and (rule.allowed is not None or rule.format == "email" or rule.case_insensitive):
                continue  # handled by allowed.noncanonical / format.email_case
            spellings: dict[str, Counter] = defaultdict(Counter)
            cells: list[tuple[int, Any, str]] = []
            for xl_row, v in self.sheet.df[col].items():
                if not isinstance(v, str) or is_null(v, self.nulls):
                    continue
                spelled = collapse_ws(v)
                spellings[spelled.casefold()][spelled] += 1
                cells.append((int(xl_row), v, spelled))
            canonical = {k: best_spelling(c) for k, c in spellings.items() if len(c) > 1}
            for xl_row, v, spelled in cells:
                canon = canonical.get(spelled.casefold())
                if canon is not None and spelled != canon:
                    self.emit("format.case_variant", "warning",
                              f"'{spelled}' is written '{canon}' elsewhere in '{col}'.", xl_row, col, v)

    def _unique_together(self, rule_id: str, cols: list[str], severity: str, message: str | None) -> None:
        missing = [c for c in cols if c not in self.sheet.df.columns]
        if missing:
            self.rule_error(rule_id, f"unknown column(s): {missing}")
            return
        first: dict[tuple, int] = {}
        df = self.sheet.df
        for xl_row, row in zip(df.index, df[cols].itertuples(index=False), strict=True):
            key = tuple(to_text(v, self.nulls) for v in row)
            if any(k is None for k in key):
                continue
            if key in first:
                label = ", ".join(key)
                self.emit(rule_id, severity, message or f"Duplicate key ({label}); first seen in row {first[key]}.",
                          int(xl_row), cols[0], row[0])
            else:
                first[key] = int(xl_row)

    # ------------------------------------------------------------------ table rules
    def _when_mask(self, trule: TableRule) -> pd.Series:
        if not trule.when:
            return pd.Series(True, index=self.sheet.df.index)
        m = self.typed.evaluate(compile_expr(trule.when))
        if m.dtype != "boolean":
            raise ExprError("'when' must be a condition")
        return m.fillna(False).astype(bool)

    def _table_rule(self, trule: TableRule) -> None:
        df = self.sheet.df
        kind = trule.kind
        if kind == "unique_together":
            self._unique_together(trule.id, trule.unique_together or [], trule.severity, trule.message)
            return
        if kind == "lookup":
            self._lookup(trule)
            return
        mask = self._when_mask(trule)
        if kind == "require":
            for col in trule.require or []:
                if col not in df.columns:
                    self.rule_error(trule.id, f"unknown column '{col}'")
                    continue
                for xl_row, v in df[col].items():
                    if mask.loc[xl_row] and is_null(v, self.nulls):
                        self.emit(trule.id, trule.severity,
                                  trule.message or f"'{col}' is required when {trule.when}.",
                                  int(xl_row), col, v)
            return
        expr = compile_expr(trule.check or "")
        res = self.typed.evaluate(expr)
        if res.dtype != "boolean":
            raise ExprError("'check' must be a condition (comparison), not a value")
        failed = mask & (res.fillna(True) == False)  # noqa: E712 - unknown is not a failure
        anchor = next((c for c in expr.columns if c in df.columns), None)
        for xl_row in df.index[failed.to_numpy()]:
            context = ", ".join(f"{c}={_fmt(df.at[xl_row, c])}" for c in expr.columns if c in df.columns)
            text = trule.message or f"Check failed: {trule.check}"
            self.emit(trule.id, trule.severity, f"{text} ({context})", int(xl_row), anchor,
                      df.at[xl_row, anchor] if anchor else None)

    def _lookup(self, trule: TableRule) -> None:
        spec = trule.lookup
        assert spec is not None
        df = self.sheet.df
        other = self.wb.sheets.get(spec.sheet)
        if spec.column not in df.columns:
            self.rule_error(trule.id, f"unknown column '{spec.column}'")
            return
        if other is None or spec.key not in other.df.columns:
            self.rule_error(trule.id, f"lookup target {spec.sheet}.{spec.key} not found")
            return
        keys = {to_text(v, self.nulls) for v in other.df[spec.key]} - {None}
        for xl_row, v in df[spec.column].items():
            t = to_text(v, self.nulls)
            if t is not None and t not in keys:
                self.emit(trule.id, trule.severity,
                          trule.message or f"'{t}' was not found in {spec.sheet}.{spec.key}.",
                          int(xl_row), spec.column, v)


def best_spelling(counter: Counter) -> str:
    """Most common spelling; ties go to Title Case, then to mixed case over ALL CAPS / lowercase."""
    def score(item: tuple[str, int]) -> tuple:
        s, n = item
        return (n, s.istitle(), not (s.isupper() or s.islower()))
    return max(counter.items(), key=score)[0]


def _as_date(b: Any) -> date | None:
    if b is None:
        return None
    if isinstance(b, date):
        return b if type(b) is date else b.date()
    raise ExprError("date bounds must be dates (YYYY-MM-DD)")


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, date):
        return v.isoformat()[:10] if not hasattr(v, "hour") or v.hour == 0 else v.isoformat()
    if v is None:
        return "null"
    return str(v)
