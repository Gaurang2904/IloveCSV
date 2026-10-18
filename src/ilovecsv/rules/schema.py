"""YAML rule specification, validated with pydantic.

Two layouts are accepted:

* single sheet: ``sheet``, ``key``, ``columns`` and ``rules`` at the top level;
* multi sheet: a ``sheets:`` list of those blocks.

Loading compiles every regex and parses every expression, so a bad rule file
fails at load time with a precise location instead of halfway through a run.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..values import DEFAULT_NULLS
from .expr import ExprError, compile_expr

Severity = Literal["error", "warning", "info"]
ColumnType = Literal["int", "float", "number", "date", "bool", "text"]
Bound = float | date | datetime


class RulesError(Exception):
    """The rule file is invalid. ``errors`` lists {loc, msg, line?, column?}."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.errors = errors or []


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnRule(_Strict):
    required: bool = False
    type: ColumnType | None = None
    min: Bound | None = None
    max: Bound | None = None
    not_future: bool = False
    allowed: list[str] | None = None
    case_insensitive: bool = False
    regex: str | None = None
    unique: bool = False
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    format: Literal["email", "phone"] | None = None
    phone_digits: int = Field(default=10, ge=4, le=15)
    # Known spellings that mean an allowed/canonical value, e.g. {Bangalore: Bengaluru, Hydrabad: Hyderabad}.
    # Matching ignores case and spacing. Typos with no correct spelling elsewhere in the data cannot be
    # detected statistically; a reference list like this is how they get fixed.
    aliases: dict[str, str] | None = None
    severity: Severity = "error"
    message: str | None = None

    @field_validator("aliases", mode="before")
    @classmethod
    def _aliases_as_strings(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return v

    @field_validator("allowed", mode="before")
    @classmethod
    def _allowed_as_strings(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [str(x) for x in v]
        return v

    @field_validator("regex")
    @classmethod
    def _regex_compiles(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"invalid regex: {e}") from e
        return v

    @model_validator(mode="after")
    def _bounds_consistent(self) -> ColumnRule:
        for name in ("min", "max"):
            b = getattr(self, name)
            if b is None:
                continue
            if self.type == "date" and not isinstance(b, date):
                raise ValueError(f"{name} must be a date (YYYY-MM-DD) for a date column")
            if self.type in ("int", "float", "number") and isinstance(b, date):
                raise ValueError(f"{name} must be a number for a numeric column")
        if self.min is not None and self.max is not None:
            try:
                if self.min > self.max:  # type: ignore[operator]
                    raise ValueError("min is greater than max")
            except TypeError as e:
                raise ValueError("min and max must be the same kind (number or date)") from e
        return self


class LookupSpec(_Strict):
    column: str
    sheet: str
    key: str


class FixSpec(_Strict):
    """A rule-declared correction: set ``column`` to the value of ``to``."""

    column: str
    to: str


class TableRule(_Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$")
    when: str | None = None
    check: str | None = None
    require: list[str] | None = None
    lookup: LookupSpec | None = None
    unique_together: list[str] | None = None
    fix: FixSpec | None = None
    severity: Severity = "error"
    message: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> TableRule:
        kinds = [k for k in ("check", "require", "lookup", "unique_together") if getattr(self, k)]
        if len(kinds) != 1:
            raise ValueError(
                "a rule needs exactly one of: check, require, lookup, unique_together "
                f"(got {kinds or 'none'})"
            )
        if self.fix and not self.check:
            raise ValueError("fix is only allowed on check rules")
        for name in ("when", "check"):
            text = getattr(self, name)
            if text is not None:
                try:
                    compile_expr(text)
                except ExprError as e:
                    raise ValueError(f"{name}: {e}") from e
        if self.fix:
            try:
                compile_expr(self.fix.to)
            except ExprError as e:
                raise ValueError(f"fix.to: {e}") from e
        return self

    @property
    def kind(self) -> str:
        for k in ("check", "require", "lookup", "unique_together"):
            if getattr(self, k):
                return k
        raise AssertionError("unreachable")


class Checks(_Strict):
    whitespace: bool = True
    duplicates: bool = True
    # Columns left out when comparing rows for duplicates. Default (None): the key columns and
    # id-like columns, so two records that differ only in their generated id still count.
    duplicate_ignore: list[str] | None = None
    # Compare only these columns (e.g. [Full Name, Email]); overrides duplicate_ignore.
    duplicate_columns: list[str] | None = None
    # "normalized": ignore case and extra spaces; "exact": values must match exactly.
    duplicate_match: Literal["normalized", "exact"] = "normalized"
    # Flag values that are spelled with different capitalisation elsewhere in the same column.
    case_variants: bool = True


class AnomalyConfig(_Strict):
    enabled: bool = True
    z_threshold: float = Field(default=3.5, gt=0)
    rare_threshold: float = Field(default=0.01, gt=0, lt=0.5)
    isolation_forest: bool = True
    if_min_rows: int = Field(default=200, ge=20)
    if_z_threshold: float = Field(default=6.0, gt=0)
    columns: list[str] | None = None  # restrict to these numeric columns
    exclude: list[str] = Field(default_factory=list)


class SheetRules(_Strict):
    sheet: str | None = None  # None = first sheet
    key: list[str] | None = None
    checks: Checks = Field(default_factory=Checks)
    columns: dict[str, ColumnRule] = Field(default_factory=dict)
    rules: list[TableRule] = Field(default_factory=list)

    @field_validator("columns", mode="before")
    @classmethod
    def _empty_column_blocks(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): ({} if val is None else val) for k, val in v.items()}
        return v

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> SheetRules:
        ids = [r.id for r in self.rules]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate rule ids: {dupes}")
        return self


class RuleSet(_Strict):
    version: Literal[1] = 1
    null_values: list[str] = Field(default_factory=lambda: list(DEFAULT_NULLS))
    dayfirst: bool = True
    anomalies: AnomalyConfig = Field(default_factory=AnomalyConfig)
    sheets: list[SheetRules] = Field(default_factory=lambda: [SheetRules()])

    @model_validator(mode="before")
    @classmethod
    def _single_sheet_layout(cls, data: Any) -> Any:
        if isinstance(data, dict) and "sheets" not in data:
            sheet_keys = {"sheet", "key", "checks", "columns", "rules"}
            sheet = {k: data[k] for k in sheet_keys if k in data}
            rest = {k: v for k, v in data.items() if k not in sheet_keys}
            rest["sheets"] = [sheet]
            return rest
        return data


def load_rules(text: str) -> RuleSet:
    """Parse and validate a YAML rule file; raise RulesError with locations."""
    if len(text) > 200_000:
        raise RulesError("Rule file is too large (max 200 KB).")
    try:
        data = yaml.safe_load(text) if text.strip() else {}
    except yaml.MarkedYAMLError as e:
        mark = e.problem_mark
        err = {"loc": [], "msg": str(e.problem or e)}
        if mark is not None:
            err["line"], err["column"] = mark.line + 1, mark.column + 1
        raise RulesError("Rule file is not valid YAML.", [err]) from e
    except yaml.YAMLError as e:
        raise RulesError("Rule file is not valid YAML.", [{"loc": [], "msg": str(e)}]) from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RulesError("Rule file must be a YAML mapping.", [{"loc": [], "msg": "expected a mapping"}])
    try:
        return RuleSet.model_validate(data)
    except ValidationError as e:
        errors = [
            {"loc": [str(p) for p in err["loc"]], "msg": err["msg"].removeprefix("Value error, ")}
            for err in e.errors()
        ]
        raise RulesError("Rule file failed validation.", errors) from e


def rules_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
