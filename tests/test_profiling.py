from openpyxl import Workbook

from ilovecsv.profiling import profile_workbook
from ilovecsv.reader import read_workbook


def _wb(tmp_path, rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "S"
    for r in rows:
        ws.append(r)
    p = tmp_path / "p.xlsx"
    wb.save(p)
    return read_workbook(p)


def test_exact_counts(tmp_path):
    wb = _wb(tmp_path, [
        ["id", "city", "amount", "joined"],
        ["A1", "Mumbai", 10, "2024-01-01"],
        ["A2", "mumbai ", 20, "2024-01-02"],
        ["A3", "N/A", "30", None],
        ["A4", "Pune", None, "2024-01-04"],
        ["A1", "Mumbai", 10, "2024-01-01"],  # exact duplicate of row 2
    ])
    p = profile_workbook(wb)["S"]
    assert p.n_rows == 5
    assert p.duplicate_rows == [6]
    city = p.columns["city"]
    assert city.n_missing == 1 and city.missing_rate == 0.2  # "N/A" counts as missing
    assert city.n_whitespace == 1
    assert city.case_variants and city.case_variants[0]["canonical"] == "Mumbai"
    amount = p.columns["amount"]
    assert amount.inferred_type == "int"
    assert amount.n_numeric_text == 1
    assert amount.numeric["min"] == 10 and amount.numeric["max"] == 30
    joined = p.columns["joined"]
    assert joined.inferred_type == "date" and joined.n_date_text == 4
    assert joined.dates == {"min": "2024-01-01", "max": "2024-01-04"}
    assert p.columns["id"].n_distinct == 4


def test_types_on_synthetic(orders_run):
    cols = orders_run.profiles["Orders"].columns
    assert cols["order_date"].inferred_type == "date"
    assert cols["quantity"].inferred_type == "int"
    assert cols["unit_price"].inferred_type == "float"
    assert cols["status"].inferred_type == "category"
    assert cols["order_id"].is_identifier
    assert not cols["quantity"].is_identifier
