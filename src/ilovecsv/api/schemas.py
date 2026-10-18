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
    approve_safe: bool = Field(
        default=False, description="Approve every pending suggestion classified 'safe' (Safe Auto-Clean).")
    include_row_deletions: bool = Field(
        default=False,
        description="Let approve_min_confidence also approve duplicate-row deletions. Off by default: "
                    "review the pairs at /duplicates and approve deletions by id.",
    )


class IssuePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["open", "confirmed_error", "dismissed"]
    note: str | None = Field(default=None, max_length=2000)


class ExportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotate: bool = Field(default=True, description="Highlight changed cells and add comments with the old value.")
    formats: list[Literal["csv", "parquet"]] = Field(default_factory=lambda: ["csv"],
                                                      description="Dataset files to write.")


class PrepareIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str | None = Field(default=None, max_length=255, description="Label column for supervised learning.")
    target_labels: list[str] | None = Field(
        default=None, max_length=100,
        description="Valid labels (e.g. ['No', 'Yes']). Variants are normalised to them; anything else is "
                    "flagged as unrecognised. Default: the labels found in the data.")
    remove_exact_duplicates: bool = Field(
        default=False, description="Let Safe Auto-Clean delete rows identical in every column (ids too).")
    formats: list[Literal["csv", "parquet"]] = Field(default_factory=lambda: ["csv"])
    rules_yaml: str | None = Field(default=None, max_length=200_000,
                                   description="A saved prepare config to reuse instead of suggested rules.")


def issue_out(r: IssueRow) -> dict[str, Any]:
    return {
        "id": r.id, "kind": r.kind, "sheet": r.sheet, "xl_row": r.xl_row, "column": r.column,
        "cell_ref": r.cell_ref, "rule_id": r.rule_id, "severity": r.severity, "value": r.value,
        "message": r.message, "detector": r.detector, "score": r.score, "status": r.status,
        "decided_by": r.decided_by, "decided_at": r.decided_at, "note": r.note,
        "resolution": r.resolution, "resolution_note": r.resolution_note, "context": r.context,
        "introduced_by_fix": bool(r.introduced_by_fix), "corrected_cell_ref": r.corrected_cell_ref,
    }


def suggestion_out(r: SuggestionRow) -> dict[str, Any]:
    return {
        "id": r.id, "issue_id": r.issue_id, "sheet": r.sheet, "action": r.action, "method": r.method,
        "confidence": r.confidence, "xl_row": r.xl_row, "column": r.column, "cell_ref": r.cell_ref,
        "old": r.old, "new": r.new, "rule_id": r.rule_id, "safety": r.safety, "status": r.status,
        "decided_by": r.decided_by, "decided_at": r.decided_at,
    }
