"""Regression tests from an external ML-preparation review.

The fixtures mirror the reviewer's test files (our own synthetic data):
* a churn table with placeholder words in numeric columns, '?' as a label, labels in mixed case,
  a negative age in a small file, and a leakage column broken by one stray row;
* a CSV built to test contradictory labels.
"""

import json
from datetime import date

import pandas as pd
import pytest
from openpyxl import Workbook

from ilovecsv.pipeline import prepare

COLUMNS = ["record_id", "email", "age", "city", "plan", "tenure_months", "monthly_spend",
           "support_tickets_90d", "last_login_date", "churn", "cancellation_date"]
CITIES = ["Delhi", "Mumbai", "Pune", "Chennai", "Jaipur", "Kolkata"]
PLANS = ["Basic", "Standard", "Premium"]


def build_ml_review_workbook(path):
    rows = []
    for i in range(1, 61):
        churned = i > 52
        rows.append([f"CUST-{i:03d}", f"customer{i:03d}@example.test", 21 + (i * 7) % 38, CITIES[i % 6],
                     PLANS[i % 3], 1 + (i * 5) % 47, [499, 799, 999, 1299, 1799][i % 5], (i * 3) % 4,
                     date(2025, 1 + i % 12, 1 + i % 27), "Yes" if churned else "No",
                     date(2025, 10, i - 40) if churned else None])
    r = {row[0]: row for row in rows}
    r["CUST-009"][2] = "unknown"          # placeholder word in a numeric column -> missing
    r["CUST-010"][2] = "N/A"              # placeholder -> missing
    r["CUST-011"][2] = -4                 # negative age in a 60-row file: must be a range error
    r["CUST-012"][2] = 151                # too old: range error
    r["CUST-019"][7] = "unknown"          # placeholder word in a numeric column -> missing
    r["CUST-042"][7] = "two"              # not a placeholder: unparseable, stays and is flagged
    r["CUST-041"][3] = "unknown"          # in a text column "unknown" may be a real category: kept
    r["CUST-024"][1] = "N/A"              # email placeholder: cleared, the value is still missing
    r["CUST-029"][9] = " ? "              # '?' label: a placeholder, so a missing label (not a class)
    r["CUST-030"][9] = None               # missing label
    for cid, label in [("CUST-007", " no "), ("CUST-011", "NO"), ("CUST-053", " YES "), ("CUST-054", " YES "),
                       ("CUST-055", " YES "), ("CUST-056", "yes")]:
        r[cid][9] = label                 # case/spacing variants; YES is more frequent than Yes
    r["CUST-057"][9] = "Maybe"            # an unknown label (only flagged when labels are declared)
    wb = Workbook()
    ws = wb.active
    ws.title = "ChurnData"
    ws.append(COLUMNS)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlreview")
    path = build_ml_review_workbook(d / "ml_review.xlsx")
    res = prepare(path, d / "out", target="churn", target_labels=["No", "Yes"])
    report = json.loads(res.report.read_text(encoding="utf-8"))
    df = pd.read_csv(res.dataset_path(), keep_default_na=False, na_values=[""])
    return res, report, df


def issues(res, rule_id):
    return {(i.cell_ref, i.value) for i in res.run.validation.issues if i.rule_id == rule_id}


def test_leakage_detected_despite_a_stray_row(result):
    _, report, _ = result
    leaks = report["ml"]["sheets"][0]["leakage_suspects"]
    assert [x["column"] for x in leaks] == ["cancellation_date"]
    sch = report["columns"][0]
    assert "cancellation_date" not in sch["features"]["datetime"]
    review = {r["column"]: r for r in sch["needs_review"]}
    assert "possible target leakage" in review["cancellation_date"]["reasons"][0]


def test_target_labels_normalised_and_unknown_flagged(result):
    res, report, df = result
    assert set(df["churn"].dropna()) == {"No", "Yes", "Maybe"}  # Maybe is kept, never guessed
    ts = report["ml"]["sheets"][0]["target_summary"]
    assert {c["label"] for c in ts["classes"]} == {"No", "Yes"}
    assert ts["unrecognised"] == [{"label": "Maybe", "count": 1}]
    assert {v for _, v in issues(res, "target.invalid_label")} == {"Maybe"}
    assert not any(s.new == "Yes" and s.old == "Maybe" for s in res.run.suggestions)


def test_question_mark_label_is_missing_not_a_class(result):
    res, report, df = result
    assert pd.isna(df.loc[df.record_id == "CUST-029", "churn"].iloc[0])
    missing = [i for i in res.run.validation.issues if i.rule_id == "target.missing"]
    assert any("placeholder '?'" in i.message for i in missing)


def test_negative_age_is_a_range_error(result):
    res, report, df = result
    assert ("ChurnData!C12", -4) in issues(res, "range")
    assert ("ChurnData!C13", 151) in issues(res, "range")
    assert df.loc[df.record_id == "CUST-011", "age"].iloc[0] == -4  # flagged, not deleted


def test_placeholder_words_only_cleared_in_typed_columns(result):
    res, _, df = result
    assert pd.isna(df.loc[df.record_id == "CUST-009", "age"].iloc[0])
    assert pd.isna(df.loc[df.record_id == "CUST-019", "support_tickets_90d"].iloc[0])
    assert df.loc[df.record_id == "CUST-041", "city"].iloc[0] == "unknown"  # a category, kept
    assert df.loc[df.record_id == "CUST-042", "support_tickets_90d"].iloc[0] == "two"  # flagged instead
    assert ("ChurnData!H43", "two") in issues(res, "type")


def test_audit_wording_for_cleared_placeholder(result):
    res, _, _ = result
    email = next(i for i in res.run.validation.issues if i.rule_id == "required" and i.cell_ref == "ChurnData!B25")
    assert email.resolution == "open"
    assert "placeholder 'N/A' was cleared" in email.resolution_note and "not effective" not in email.resolution_note
    token = next(i for i in res.run.validation.issues if i.rule_id == "format.null_token" and i.cell_ref == "ChurnData!B25")
    assert token.resolution == "resolved"


def test_unparseable_columns_need_review(result):
    _, report, _ = result
    sch = report["columns"][0]
    review = {r["column"] for r in sch["needs_review"]}
    assert "support_tickets_90d" in review
    assert "support_tickets_90d" not in sch["features"]["numeric"]
    assert {c["name"]: c["status"] for c in sch["columns"]}["support_tickets_90d"] == "review"


def test_conflicting_labels_csv(tmp_path):
    """Identical features with contradictory labels; case/spacing differences do not hide a conflict;
    rows differing in one feature are not conflicts; duplicates with the same label are not either."""
    lines = ["id,age,plan,city,churn"]
    base = [(30, "Basic", "Pune"), (41, "Premium", "Delhi"), (25, "Standard", "Jaipur"), (52, "Basic", "Mumbai")]
    n = 0
    for k in range(40):  # filler rows, all different
        n += 1
        lines.append(f"R{n:03d},{20 + k},{PLANS[k % 3]},{CITIES[k % 6]},{'Yes' if k % 4 == 0 else 'No'}")
    for (age, plan, city), labels in zip(base, [("Yes", "No"), ("No", "Yes", "No"), ("No", "No"), ("Yes", "No")],
                                         strict=True):
        for j, label in enumerate(labels):
            n += 1
            shown_city = city.upper() + " " if j == 1 else city  # formatting differences must not hide it
            lines.append(f"R{n:03d},{age if label != 'X' else age + 1},{plan},{shown_city},{label}")
    n += 1
    lines.append(f"R{n:03d},53,Basic,Mumbai,Yes")  # differs from the last group by age: not a conflict
    path = tmp_path / "conflicts.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = prepare(path, tmp_path / "out", target="churn")
    groups = res.run.validation.issues
    conflict_rows = sorted({tuple((i.context or {})["rows"]) for i in groups if i.rule_id == "target.conflict"})
    assert len(conflict_rows) == 3  # (Yes/No), (No/Yes/No), (Yes/No); the (No/No) pair agrees
    assert all(len(g) >= 2 for g in conflict_rows)
    report = json.loads(res.report.read_text(encoding="utf-8"))
    assert len(report["ml"]["sheets"][0]["label_conflicts"]) == 3
