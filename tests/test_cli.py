import json

import pandas as pd

from ilovecsv.cli import main
from ilovecsv.rules.expr import compile_expr


def test_check_without_export(orders, rules_text, tmp_path, capsys):
    rules = tmp_path / "rules.yaml"
    rules.write_text(rules_text, encoding="utf-8")
    code = main(["check", str(orders[0]), "--rules", str(rules), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 1  # errors present -> non-zero exit, usable in CI pipelines
    assert "quality score" in out and "total_matches" in out
    assert (tmp_path / "out" / "orders_cleaning_report.html").exists()
    assert not (tmp_path / "out" / "orders_cleaned.xlsx").exists()


def test_check_with_export(orders, rules_text, tmp_path, capsys):
    rules = tmp_path / "rules.yaml"
    rules.write_text(rules_text, encoding="utf-8")
    main(["check", str(orders[0]), "--rules", str(rules), "--out", str(tmp_path / "o"), "--approve-min", "0.9"])
    assert "corrected file" in capsys.readouterr().out
    assert (tmp_path / "o" / "orders_cleaned.xlsx").exists()


def test_profile_and_suggest(clean_orders, capsys):
    assert main(["profile", str(clean_orders)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["name"] == "Orders"
    assert main(["suggest-rules", str(clean_orders)]) == 0
    assert "columns:" in capsys.readouterr().out


def test_bad_inputs_exit_2(tmp_path, capsys, orders):
    assert main(["profile", str(tmp_path / "x.xls")]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("columns:\n  a: {regex: '('}\n")
    assert main(["check", str(orders[0]), "--rules", str(bad)]) == 2
    assert "invalid regex" in capsys.readouterr().err


def test_create_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ILOVECSV_DATA_DIR", str(tmp_path / "d"))
    assert main(["create-key", "carol"]) == 0
    assert "ilc_" in capsys.readouterr().out


def test_expression_functions():
    idx = pd.Index([1, 2])
    cols = {
        "s": pd.Series([" Ab ", None], index=idx, dtype=object),
        "x": pd.Series([1.26, None], index=idx),
        "d1": pd.to_datetime(pd.Series(["2024-01-01", "2024-01-10"], index=idx)),
        "d2": pd.to_datetime(pd.Series(["2024-01-03", "2024-01-01"], index=idx)),
    }

    def ev(t):
        return compile_expr(t).evaluate(lambda c: cols[c], idx).tolist()

    assert ev("lower(strip(s)) == 'ab'") == [True, pd.NA]
    assert ev("upper(s) == ' AB '") == [True, pd.NA]
    assert ev("len(s) == 4") == [True, pd.NA]
    assert ev("round(x, 1) == 1.3") == [True, pd.NA]
    assert ev("coalesce(x, 0) >= 0") == [True, True]
    assert ev("days_between(d1, d2) == 2") == [True, False]
    assert ev("d1 + 2 == d2") == [True, False]
    assert ev("-x < 0") == [True, pd.NA]
    assert ev("x % 1 > 0.2") == [True, pd.NA]
    assert ev("x / 0 > 1") == [pd.NA, pd.NA]
    assert ev("not_null(s) or is_null(x)") == [True, True]
