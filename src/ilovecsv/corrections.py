"""Safe correction workflow: suggest fixes, apply only what a person approved.

Suggestions are generated for *mechanical* problems where the intended value is
clear (extra spaces, a category written in the wrong case, numbers or dates
stored as text, Excel serial dates, exact duplicate rows, or a fix expression
declared in the rule file). Missing values are never imputed, out-of-range
values are never guessed, and anomalies never get suggestions.

Applying works on a copy of the original file with openpyxl, so formatting,
other sheets and formulas are kept. Each change is highlighted and gets a cell
comment with the old value, the rule, the method and the approver.
"""

from __future__ import annotations

import csv
import difflib
import io
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import PatternFill
from openpyxl.utils.cell import column_index_from_string, coordinate_from_string

from .models import APPROVED, ERROR, Issue, Suggestion, jsonable
from .profiling import SheetProfile, collapse_ws
from .reader import WorkbookData, sha256_file
from .rules.engine import TypedColumns, best_spelling
from .rules.expr import ExprError, compile_expr
from .rules.schema import RuleSet, SheetRules
from .values import (
    excel_serial_to_date,
    normalise_nulls,
    normalize_phone,
    parse_date_text,
    to_bool,
    to_number,
    to_text,
)

CHANGED_FILL = PatternFill("solid", fgColor="FFF2CC")
KEPT_DUPLICATE_FILL = PatternFill("solid", fgColor="F8CBAD")

# When several suggestions target the same cell, the first in this list wins.
_PRIORITY = [
    "rule_fix",
    "apply_alias",
    "fuzzy_category",
    "canonical_category",
    "canonical_spelling",
    "lowercase_email",
    "normalize_phone",
    "text_to_date",
    "text_to_date_ambiguous",
    "serial_to_date",
    "text_to_number",
    "text_to_bool",
    "trim_whitespace",
]


def suggest_corrections(
    wb: WorkbookData, rules: RuleSet, issues: list[Issue], profiles: dict[str, SheetProfile]
) -> list[Suggestion]:
    nulls = normalise_nulls(rules.null_values)
    by_sheet: dict[str, SheetRules] = {(s.sheet or wb.first_sheet.name): s for s in rules.sheets}
    typed: dict[str, TypedColumns] = {}
    spellings: dict[tuple[str, str | None], dict[str, str]] = {}
    out: list[Suggestion] = []

    for issue in issues:
        if issue.kind != ERROR or issue.sheet not in wb.sheets:
            continue
        sheet = wb.sheets[issue.sheet]
        srules = by_sheet.get(issue.sheet, SheetRules())
        col_rule = srules.columns.get(issue.column) if issue.column else None
        raw = sheet.df.at[issue.xl_row, issue.column] if issue.xl_row and issue.column else None
        if issue.xl_row and issue.column and sheet.is_formula(issue.xl_row, issue.column):
            continue

        def sug(method: str, new: Any, confidence: float, *, column: str | None = None,
                issue: Issue = issue, sheet=sheet) -> None:
            column = column or issue.column
            old = sheet.df.at[issue.xl_row, column]
            out.append(Suggestion(issue_id=issue.id, sheet=issue.sheet, action="set", method=method,
                                  confidence=round(confidence, 3), xl_row=issue.xl_row, column=column,
                                  cell_ref=sheet.cell_ref(issue.xl_row, column), old=old, new=new,
                                  rule_id=issue.rule_id))

        rid = issue.rule_id
        if rid == "format.whitespace" and isinstance(raw, str):
            sug("trim_whitespace", collapse_ws(raw), 0.99)
        elif rid == "format.email_case" and isinstance(raw, str):
            sug("lowercase_email", raw.strip().lower(), 0.95)
        elif rid == "format.phone_style" and col_rule:
            norm = normalize_phone(to_text(raw, nulls) or "", col_rule.phone_digits)
            if norm is not None:
                sug("normalize_phone", norm, 0.95)
        elif rid == "allowed.alias" and col_rule and col_rule.aliases:
            folded = {collapse_ws(k).casefold(): v for k, v in col_rule.aliases.items()}
            canonical = folded.get(collapse_ws(str(raw)).casefold())
            if canonical is not None:
                sug("apply_alias", canonical, 0.95)
        elif rid == "format.case_variant" and isinstance(raw, str):
            key = (issue.sheet, issue.column)
            if key not in spellings:
                groups: dict[str, Counter] = defaultdict(Counter)
                for v in sheet.df[issue.column]:
                    if isinstance(v, str) and v.strip():
                        groups[collapse_ws(v).casefold()][collapse_ws(v)] += 1
                spellings[key] = {k: best_spelling(c) for k, c in groups.items()}
            canon = spellings[key].get(collapse_ws(raw).casefold())
            if canon is not None:
                sug("canonical_spelling", canon, 0.9)
        elif rid == "allowed.noncanonical" and col_rule and col_rule.allowed:
            folded = {collapse_ws(a).casefold(): a for a in col_rule.allowed}
            canonical = folded.get(collapse_ws(str(raw)).casefold())
            if canonical is not None:
                sug("canonical_category", canonical, 0.97)
        elif rid == "allowed" and col_rule and col_rule.allowed:
            match = closest_category(str(raw), col_rule.allowed)
            if match:
                sug("fuzzy_category", match[0], match[1])
        elif rid == "type.stored_as_text" and col_rule:
            if col_rule.type == "bool":
                b = to_bool(raw, nulls)
                if b is not None:
                    sug("text_to_bool", b, 0.95)
            else:
                num = to_number(raw, nulls)
                if num is not None:
                    new = int(num) if col_rule.type == "int" or num.is_integer() else num
                    sug("text_to_number", new, 0.95)
        elif rid == "type.date_as_text" and isinstance(raw, str):
            d, ambiguous = parse_date_text(raw, rules.dayfirst)
            if d is not None:
                if ambiguous:
                    sug("text_to_date_ambiguous", d, 0.5)
                else:
                    sug("text_to_date", d, 0.95)
        elif rid == "type.date_serial" and isinstance(raw, (int, float)):
            sug("serial_to_date", excel_serial_to_date(raw), 0.9)
        elif rid == "row.duplicate" and issue.xl_row:
            out.append(Suggestion(issue_id=issue.id, sheet=issue.sheet, action="delete_row",
                                  method="remove_duplicate_row", confidence=0.9, xl_row=issue.xl_row,
                                  cell_ref=sheet.row_ref(issue.xl_row), old=issue.message, new=None,
                                  rule_id=rid))
        else:
            trule = next((r for r in srules.rules if r.id == rid and r.fix), None)
            if trule and trule.fix and trule.fix.column in sheet.df.columns and issue.xl_row:
                if sheet.is_formula(issue.xl_row, trule.fix.column):
                    continue
                if issue.sheet not in typed:
                    typed[issue.sheet] = TypedColumns(sheet, srules, profiles[issue.sheet], nulls, rules.dayfirst)
                try:
                    values = typed[issue.sheet].evaluate(compile_expr(trule.fix.to))
                except ExprError:
                    continue
                v = values.loc[issue.xl_row]
                if v is None or (isinstance(v, float) and v != v):
                    continue
                if hasattr(v, "item"):
                    v = v.item()
                if isinstance(v, float):
                    v = round(v, 6)
                    if v.is_integer():
                        v = int(v)
                sug("rule_fix", v, 0.8, column=trule.fix.column)
    return _one_per_cell(out)


def closest_category(value: str, allowed: list[str], cutoff: float = 0.8) -> tuple[str, float] | None:
    """Best fuzzy match, only when it is clearly better than the runner-up."""
    v = collapse_ws(value).casefold()
    scored = sorted(
        ((difflib.SequenceMatcher(None, v, collapse_ws(a).casefold()).ratio(), a) for a in allowed),
        reverse=True,
    )
    if not scored or scored[0][0] < cutoff:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.1:
        return None
    return scored[0][1], scored[0][0]


def _one_per_cell(suggestions: list[Suggestion]) -> list[Suggestion]:
    best: dict[str, Suggestion] = {}
    order = {m: i for i, m in enumerate(_PRIORITY)}
    out: list[Suggestion] = []
    for s in suggestions:
        if s.action != "set":
            out.append(s)
            continue
        cur = best.get(s.cell_ref or "")
        if cur is None or order.get(s.method, 99) < order.get(cur.method, 99):
            best[s.cell_ref or ""] = s
    return out + list(best.values())


# ----------------------------------------------------------------------------- apply


@dataclass
class ApplyResult:
    out_path: Path
    sha256: str
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def apply_corrections(
    wb: WorkbookData,
    suggestions: list[Suggestion],
    out_path: str | Path,
    annotate: bool = True,
) -> ApplyResult:
    """Write a corrected copy containing only APPROVED suggestions."""
    out_path = Path(out_path)
    approved = [s for s in suggestions if s.status == APPROVED]
    if wb.file_type == "csv":
        applied, skipped = _apply_csv(wb, approved, out_path)
    else:
        applied, skipped = _apply_xlsx(wb, approved, out_path, annotate)
    return ApplyResult(out_path, sha256_file(out_path), applied, skipped)


def _record(s: Suggestion) -> dict[str, Any]:
    return {"suggestion_id": s.id, "issue_id": s.issue_id, "cell_ref": s.cell_ref, "action": s.action,
            "old": s.old, "new": s.new, "method": s.method, "rule_id": s.rule_id,
            "approved_by": s.decided_by, "approved_at": s.decided_at}


def _apply_xlsx(wb: WorkbookData, approved: list[Suggestion], out_path: Path, annotate: bool):
    book = load_workbook(wb.path)  # formulas kept as formulas
    applied, skipped = [], []
    has_formulas = any(s.formula_cells for s in wb.sheets.values())

    for s in (x for x in approved if x.action == "set"):
        ws = book[s.sheet]
        coord = (s.cell_ref or "").split("!", 1)[1]
        cell = ws[coord]
        if cell.data_type == "f":
            skipped.append({**_record(s), "reason": "formula cell; not overwritten"})
            continue
        if jsonable(cell.value) != s.old:
            skipped.append({**_record(s), "reason": "cell changed since validation"})
            continue
        cell.value = s.new_raw
        if isinstance(s.new_raw, (date, datetime)):
            cell.number_format = "yyyy-mm-dd"
        if annotate:
            cell.fill = CHANGED_FILL
            cell.comment = _comment(f"IloveCSV changed this cell\nwas: {s.old!r}\nrule: {s.rule_id}\n"
                                    f"method: {s.method}\napproved by: {s.decided_by or '-'}")
        applied.append(_record(s))

    deletes = sorted((x for x in approved if x.action == "delete_row"), key=lambda x: -(x.xl_row or 0))
    for s in deletes:
        ws = book[s.sheet]
        header = wb.sheets[s.sheet].header_row
        merged_below = any(m.min_row > header for m in ws.merged_cells.ranges)
        if has_formulas or merged_below:
            reason = ("workbook contains formulas" if has_formulas else "sheet has merged cells") + \
                     "; deleting rows could break references, so the row was kept and marked"
            if annotate:
                for c in ws[s.xl_row]:
                    c.fill = KEPT_DUPLICATE_FILL
                ws.cell(row=s.xl_row, column=1).comment = _comment(
                    f"IloveCSV: duplicate row approved for removal but kept ({reason}).")
            skipped.append({**_record(s), "reason": reason})
            continue
        ws.delete_rows(s.xl_row)
        applied.append(_record(s))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    book.save(out_path)
    return applied, skipped


def _comment(text: str) -> Comment:
    c = Comment(text, "IloveCSV")
    c.width, c.height = 260, 110
    return c


def _apply_csv(wb: WorkbookData, approved: list[Suggestion], out_path: Path):
    raw = wb.path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    newline = "\r\n" if b"\r\n" in raw[:65536] else "\n"
    delimiter = wb.csv_delimiter or ","
    grid = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    applied, skipped = [], []
    for s in (x for x in approved if x.action == "set"):
        col_letter, row = coordinate_from_string((s.cell_ref or "").split("!", 1)[1])
        c = column_index_from_string(col_letter) - 1
        r = row - 1
        current = grid[r][c] if r < len(grid) and c < len(grid[r]) else None
        if (current or None) != s.old:
            skipped.append({**_record(s), "reason": "cell changed since validation"})
            continue
        grid[r][c] = _csv_text(s.new_raw)
        applied.append(_record(s))
    for s in sorted((x for x in approved if x.action == "delete_row"), key=lambda x: -(x.xl_row or 0)):
        del grid[(s.xl_row or 1) - 1]
        applied.append(_record(s))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=delimiter, lineterminator=newline).writerows(grid)
    return applied, skipped


def _csv_text(v: Any) -> str:
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return "" if v is None else str(v)
