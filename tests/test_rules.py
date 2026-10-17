"""Does each rule catch exactly the cells that were broken on purpose?"""

from collections import defaultdict

import pytest

from ilovecsv.pipeline import run_checks
from ilovecsv.synth import build_orders_workbook

EXPECTED_RULES = [
    "required", "regex", "range", "not_future", "type", "type.date_as_text", "type.date_serial",
    "type.stored_as_text", "allowed", "allowed.noncanonical", "format.whitespace", "ship_after_order",
    "shipped_needs_date", "total_matches", "customer_exists", "row.duplicate", "key.duplicate",
]  # fmt: skip


def _by_rule(pairs):
    out = defaultdict(set)
    for ref, rule in pairs:
        out[rule].add(ref)
    return out


@pytest.fixture(scope="module")
def truth_and_found(orders, orders_run):
    _, manifest = orders
    truth = _by_rule((m["cell_ref"], m["rule_id"]) for m in manifest if m["kind"] == "error")
    found = _by_rule((i.cell_ref, i.rule_id) for i in orders_run.validation.issues)
    return truth, found


def test_manifest_covers_every_rule_type(truth_and_found):
    truth, _ = truth_and_found
    assert set(EXPECTED_RULES) <= set(truth)


@pytest.mark.parametrize("rule_id", EXPECTED_RULES)
def test_rule_catches_exactly_injected_cells(truth_and_found, rule_id):
    truth, found = truth_and_found
    assert found[rule_id] == truth[rule_id], (
        f"missed={sorted(truth[rule_id] - found[rule_id])} extra={sorted(found[rule_id] - truth[rule_id])}"
    )


def test_no_unexpected_rules_fire(truth_and_found):
    truth, found = truth_and_found
    assert set(found) <= set(truth)


def test_no_rule_errors(orders_run):
    assert orders_run.validation.rule_errors == []


def test_clean_workbook_has_no_issues(clean_orders, rules_text):
    run = run_checks(clean_orders, rules_text)
    assert run.validation.issues == []
    assert run.anomalies == []
    assert run.summary()["quality_score"] == 100.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_other_seeds_also_exact(tmp_path, rules_text, seed):
    path = tmp_path / f"o{seed}.xlsx"
    manifest = build_orders_workbook(path, seed=seed)
    run = run_checks(path, rules_text)
    truth = {(m["cell_ref"], m["rule_id"]) for m in manifest if m["kind"] == "error"}
    found = {(i.cell_ref, i.rule_id) for i in run.validation.issues}
    assert found == truth


def test_missing_column_and_sheet(orders, rules_text):
    text = rules_text + "  - sheet: Nope\n    columns: {x: {required: true}}\n"
    text = text.replace("      email:", "      fax: {required: true}\n      email:")
    run = run_checks(orders[0], text)
    rules = {(i.rule_id, i.column) for i in run.validation.issues}
    assert ("column_missing", "fax") in rules
    assert any(i.rule_id == "sheet_missing" and i.sheet == "Nope" for i in run.validation.issues)


def test_broken_rule_reported_not_crashing(orders):
    text = "sheet: Orders\nrules:\n  - id: bad\n    check: 'order_date > 5'\n"
    run = run_checks(orders[0], text)
    assert run.validation.rule_errors and run.validation.rule_errors[0]["rule_id"] == "bad"


def test_messages_point_at_cells(orders_run):
    for i in orders_run.validation.issues:
        assert i.message
        if i.xl_row is not None:
            assert i.cell_ref and i.cell_ref.startswith(f"{i.sheet}!")
