import pandas as pd
import pytest

from ilovecsv.rules.expr import ExprError, compile_expr
from ilovecsv.rules.schema import RulesError, load_rules

IDX = pd.Index([2, 3, 4])
COLS = {
    "a": pd.Series([1.0, 5.0, None], index=IDX),
    "b": pd.Series([2.0, 3.0, 4.0], index=IDX),
    "d1": pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01", None], index=IDX)),
    "d2": pd.to_datetime(pd.Series(["2024-01-05", "2024-01-01", "2024-01-01"], index=IDX)),
    "s": pd.Series(["x", None, "y"], index=IDX, dtype=object),
    "Unit Price": pd.Series([1.0, 2.0, 3.0], index=IDX),
}


def ev(text):
    return compile_expr(text).evaluate(lambda c: COLS[c], IDX).tolist()


def test_null_is_unknown_not_false():
    assert ev("a < b") == [True, False, pd.NA]
    assert ev("a is null") == [False, False, True]
    assert ev("a is not null and a < b") == [True, False, False]


def test_dates_and_arithmetic():
    assert ev("d2 >= d1") == [True, False, pd.NA]
    assert ev("d2 - d1 <= 7") == [True, True, pd.NA]
    assert ev("d1 < date('2024-01-15')") == [True, False, pd.NA]
    assert ev("abs(a - b) <= 1") == [True, False, pd.NA]
    assert ev("`Unit Price` * 2 > 3") == [False, True, True]
    assert ev("s in ['x', 'z']") == [True, pd.NA, False]


@pytest.mark.parametrize("bad", [
    "__import__('os').system('x')", "a.real", "open('f')", "[x for x in a]", "a[0]", "lambda: 1",
    "a is 3", "s in a", "(1).__class__", "eval('1')", "__builtins__", "a if b else 1", "x := 1",
])
def test_rejects_unsafe(bad):
    with pytest.raises(ExprError):
        compile_expr(bad)


def test_columns_listed():
    assert compile_expr("abs(total - qty * `unit price`) <= 0.01").columns == ["total", "qty", "unit price"]


def test_yaml_error_has_line():
    with pytest.raises(RulesError) as e:
        load_rules("columns:\n  a: {required: true\n  b: {}\n")
    assert e.value.errors[0]["line"] >= 2


@pytest.mark.parametrize("text,needle", [
    ("columns:\n  a: {regex: '('}\n", "invalid regex"),
    ("columns:\n  a: {min: 5, max: 1}\n", "min is greater than max"),
    ("columns:\n  a: {type: date, min: 3}\n", "must be a date"),
    ("columns:\n  a: {requird: true}\n", "Extra inputs"),
    ("rules:\n  - id: r1\n    check: 'a.b > 1'\n", "not allowed"),
    ("rules:\n  - id: r1\n", "exactly one of"),
    ("rules:\n  - {id: r1, check: 'a > 1'}\n  - {id: r1, check: 'a > 2'}\n", "duplicate rule ids"),
])
def test_rule_validation_messages(text, needle):
    with pytest.raises(RulesError) as e:
        load_rules(text)
    assert any(needle in err["msg"] for err in e.value.errors), e.value.errors


def test_single_and_multi_sheet_layouts(rules_text):
    single = load_rules("sheet: S\ncolumns:\n  a: {required: true}\n")
    assert single.sheets[0].sheet == "S" and single.sheets[0].columns["a"].required
    multi = load_rules(rules_text)
    assert [s.sheet for s in multi.sheets] == ["Orders", "Customers"]
