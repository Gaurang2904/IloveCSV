from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..db import IssueRow, SuggestionRow


class RulesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules_yaml: str | None = Field(
        default=None, max_length=200_000,
        description="Rule file as YAML text. Omit to validate with rules suggested from the data profile.",
    )


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: list[str] = Field(default_factory=list, max_length=10_000)
    reject: list[str] = Field(default_factory=list, max_length=10_000)
    approve_min_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Also approve every pending suggestion at or above this confidence."
    )
    rule_id: str | None = Field(default=None, description="Limit approve_min_confidence to one rule.")
    method: str | None = Field(default=None, description="Limit approve_min_confidence to one method.")


class IssuePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["open", "confirmed_error", "dismissed"]
    note: str | None = Field(default=None, max_length=2000)


class ExportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotate: bool = Field(default=True, description="Highlight changed cells and add comments with the old value.")


def issue_out(r: IssueRow) -> dict[str, Any]:
    return {
        "id": r.id, "kind": r.kind, "sheet": r.sheet, "xl_row": r.xl_row, "column": r.column,
        "cell_ref": r.cell_ref, "rule_id": r.rule_id, "severity": r.severity, "value": r.value,
        "message": r.message, "detector": r.detector, "score": r.score, "status": r.status,
        "decided_by": r.decided_by, "decided_at": r.decided_at, "note": r.note,
    }


def suggestion_out(r: SuggestionRow) -> dict[str, Any]:
    return {
        "id": r.id, "issue_id": r.issue_id, "sheet": r.sheet, "action": r.action, "method": r.method,
        "confidence": r.confidence, "xl_row": r.xl_row, "column": r.column, "cell_ref": r.cell_ref,
        "old": r.old, "new": r.new, "rule_id": r.rule_id, "status": r.status,
        "decided_by": r.decided_by, "decided_at": r.decided_at,
    }
