"""Shared result types: issues, suggestions and their JSON form."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any

ERROR = "error"
ANOMALY = "anomaly"

# Issue status values. Errors start "open"; anomalies can be confirmed or dismissed.
OPEN = "open"
CONFIRMED_ERROR = "confirmed_error"
DISMISSED = "dismissed"
ISSUE_STATUSES = {OPEN, CONFIRMED_ERROR, DISMISSED}

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
SUGGESTION_STATUSES = {PENDING, APPROVED, REJECTED}

# Issue resolutions.
#   resolved        a correction was applied and the cell/row passes re-validation
#   open            unresolved; no automatic correction exists, so a person must fix the data
#   pending_review  a person must decide: a suggested fix not yet approved, or an anomaly not reviewed
#   approved        fix approved, not exported yet (becomes resolved or open after export)
#   rejected        the suggested fix was rejected, so the issue remains
#   dismissed       an anomaly the reviewer judged to be correct data
RESOLVED = "resolved"
PENDING_REVIEW = "pending_review"
DISMISSED_RES = "dismissed"
RESOLUTIONS = (RESOLVED, OPEN, PENDING_REVIEW, APPROVED, REJECTED, DISMISSED_RES)


def stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic id, so the same input and rules always give the same ids."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def jsonable(v: Any) -> Any:
    """Convert a cell value into something json.dumps accepts."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (datetime, date, time)):
        return v.isoformat()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):  # numpy scalar
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


@dataclass
class Issue:
    kind: str  # "error" (rule violated) or "anomaly" (statistically suspicious)
    sheet: str
    rule_id: str
    severity: str  # "error" | "warning" | "info"
    message: str
    xl_row: int | None = None
    column: str | None = None
    cell_ref: str | None = None  # e.g. "Orders!D14"
    value: Any = None
    detector: str | None = None  # anomalies only
    score: float | None = None  # anomalies only
    status: str = OPEN  # anomaly review: open | confirmed_error | dismissed
    # What happened to the issue: see RESOLUTIONS. Set from decisions, then finally from re-validation.
    resolution: str = ""
    resolution_note: str | None = None
    context: dict[str, Any] | None = None  # rule-specific details, e.g. {"duplicate_of": 2}
    id: str = ""

    def __post_init__(self) -> None:
        self.value = jsonable(self.value)
        if not self.id:
            self.id = stable_id(self.kind[0], self.sheet, self.xl_row, self.column, self.rule_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Suggestion:
    issue_id: str
    sheet: str
    action: str  # "set" | "delete_row"
    method: str
    confidence: float
    xl_row: int | None = None
    column: str | None = None
    cell_ref: str | None = None
    old: Any = None
    new: Any = None
    rule_id: str = ""
    status: str = PENDING
    decided_by: str | None = None
    decided_at: str | None = None
    # "safe": lossless or pure formatting, applied by Safe Auto-Clean.
    # "review": ambiguous, inferred or destructive; a person decides.
    safety: str = "review"
    id: str = ""
    # The typed new value used when writing the workbook (e.g. a real date).
    new_raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.new_raw is None:
            self.new_raw = self.new
        self.old = jsonable(self.old)
        self.new = jsonable(self.new)
        if not self.id:
            self.id = stable_id("s", self.issue_id, self.method)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("new_raw", None)
        return d
