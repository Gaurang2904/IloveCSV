from datetime import date, datetime

import pytest

from ilovecsv.reader import ReadError, read_workbook
from ilovecsv.values import (
    excel_serial_to_date,
    parse_date_text,
    parse_number_text,
    to_number,
    to_text,
    value_kind,
)


@pytest.mark.parametrize("raw,expected", [
    ("1,200.50", 1200.5), ("₹ 1,000", 1000.0), ("-3", -3.0), (".5", 0.5), ("12a", None), ("1,20", None),
])
def test_parse_number_text(raw, expected):
    assert parse_number_text(raw) == expected


def test_date_text_ambiguity():
    assert parse_date_text("15/03/2024") == (date(2024, 3, 15), False)
    assert parse_date_text("03/04/2024", dayfirst=True) == (date(2024, 4, 3), True)
    assert parse_date_text("03/04/2024", dayfirst=False) == (date(2024, 3, 4), True)
    assert parse_date_text("2024-02-30") == (None, False)


def test_value_kinds_keep_identifiers_as_text():
    assert value_kind("00123") == "text"
    assert value_kind("12") == "numeric_text"
    assert value_kind(" N/A ") == "null"
    assert value_kind(3.0) == "int"
    assert value_kind(datetime(2024, 1, 1)) == "date"
    assert value_kind("yes") == "bool_text"
    assert to_number(True) is None
    assert to_text(5.0) == "5"


def test_excel_serial():
    assert excel_serial_to_date(45000) == date(2023, 3, 15)


def test_edge_workbook_layout(edge_xlsx):
    wb = read_workbook(edge_xlsx)
    assert list(wb.sheets) == ["Data"]  # the empty sheet is skipped...
    assert any("Empty" in w for w in wb.warnings)  # ...and reported
    s = wb.sheets["Data"]
    assert s.header_row == 3  # title row and blank row skipped
    assert list(s.df.columns) == ["ID", "Name", "Amount", "Amount.1", "Joined", "Double"]  # merged header spread
    assert list(s.df.index) == [4, 5, 7]  # Excel row numbers kept
    assert s.dropped_blank_rows == [6]
    assert s.df.at[4, "ID"] == "00123"  # leading zeros survive
    assert s.cell_ref(7, "Amount") == "Data!C7"
    assert {"F4", "F5", "F7"} <= s.formula_cells


def test_csv_roundtrip(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("id;amount\n001;1,5\n002;3\n", encoding="utf-8")
    wb = read_workbook(p)
    assert wb.csv_delimiter == ";"
    s = wb.first_sheet
    assert list(s.df.columns) == ["id", "amount"]
    assert s.df.at[2, "id"] == "001"


def test_rejects_unsupported(tmp_path):
    p = tmp_path / "x.xls"
    p.write_bytes(b"whatever")
    with pytest.raises(ReadError):
        read_workbook(p)
    q = tmp_path / "fake.xlsx"
    q.write_text("not a zip")
    with pytest.raises(ReadError):
        read_workbook(q)
