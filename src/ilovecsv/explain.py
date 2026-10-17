"""Plain-English explanations of issue types.

Deterministic templates for now. This is the single place an LLM could later be
plugged in to explain a specific failure in context; the rest of the system only
calls ``explain_issue``.
"""

from __future__ import annotations

from .models import Issue

WHY = {
    "required": "A required field is empty, so downstream reports and joins will miss this record.",
    "type": "The value cannot be read as the expected type, so calculations or sorting on this column will fail or skip it.",
    "type.stored_as_text": "The value is a number or boolean stored as text; Excel and most tools will not sum or compare it correctly.",
    "type.date_as_text": "The date is stored as text, so date filters, sorting and arithmetic will treat it as a string.",
    "type.date_serial": "An Excel serial number sits where a date is expected; it is probably a date whose formatting was lost.",
    "range": "The value is outside the range the rules allow, which usually means a typo or a unit mistake.",
    "not_future": "The date lies in the future, which is impossible for this field.",
    "allowed": "The value is not in the list of allowed categories, so it will be miscounted in any grouping.",
    "allowed.noncanonical": "The category is valid but spelled with different case or spacing, which splits one group into several.",
    "regex": "The value does not follow the required format.",
    "length": "The text is shorter or longer than allowed.",
    "unique": "The value must be unique but appears more than once.",
    "key.duplicate": "Two rows share the same key, so lookups on this key are ambiguous.",
    "row.duplicate": "The whole row repeats an earlier row; totals that include both will double-count.",
    "format.whitespace": "Leading, trailing or repeated spaces make equal values look different to filters and lookups.",
    "column_missing": "A column the rules expect is not in the sheet.",
    "sheet_missing": "A sheet the rules expect is not in the workbook.",
    "anomaly.robust_z": "Statistically far from the rest of the column. It may be correct, so a person should confirm or dismiss it.",
    "anomaly.iqr": "Outside the far fences of the column's distribution. It may be correct, so a person should confirm or dismiss it.",
    "anomaly.rare_category": "This category is very rare in the column and might be a typo. Confirm or dismiss it.",
    "anomaly.isolation_forest": "The combination of values in this row is unusual compared with the other rows. Confirm or dismiss it.",
}


def explain_issue(issue: Issue) -> str:
    """One-sentence explanation of why the issue matters."""
    if issue.rule_id in WHY:
        return WHY[issue.rule_id]
    if issue.kind == "error":
        return "A cross-column business rule from the rule file was not satisfied."
    return "Flagged as unusual. A person should review it."
