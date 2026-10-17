"""Anomalies: injected outliers found, clean data quiet, never mixed up with errors."""

import pytest

from ilovecsv.models import ANOMALY, ERROR
from ilovecsv.pipeline import run_checks
from ilovecsv.synth import build_orders_workbook


def _row(ref: str) -> str:
    return ref.split("!")[1].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _truth(manifest):
    return {(_row(m["cell_ref"]), m["rule_id"]) for m in manifest if m["kind"] == "anomaly"}


def _found(run):
    return {(_row(i.cell_ref), i.rule_id) for i in run.anomalies}


def test_detectors_find_injected(orders, orders_run):
    truth, found = _truth(orders[1]), _found(orders_run)
    recall = len(truth & found) / len(truth)
    assert recall >= 0.9, f"missed {truth - found}"
    assert found <= truth, f"unexpected {found - truth}"
    assert {d.split(".")[1] for _, d in truth} == {"robust_z", "isolation_forest", "rare_category"}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_detectors_across_seeds(tmp_path, rules_text, seed):
    path = tmp_path / "o.xlsx"
    manifest = build_orders_workbook(path, seed=seed)
    run = run_checks(path, rules_text)
    truth, found = _truth(manifest), _found(run)
    assert len(truth & found) / len(truth) >= 0.9
    assert found <= truth


def test_clean_data_has_no_anomalies(clean_orders, rules_text):
    assert run_checks(clean_orders, rules_text).anomalies == []


def test_anomalies_are_not_errors(orders_run):
    error_cells = {i.cell_ref for i in orders_run.validation.issues}
    for a in orders_run.anomalies:
        assert a.kind == ANOMALY and a.severity == "info" and a.status == "open"
        assert a.cell_ref not in error_cells
        assert a.detector and a.score is not None
    assert all(i.kind == ERROR for i in orders_run.validation.issues)


def test_anomalies_never_get_suggestions(orders_run):
    anomaly_ids = {a.id for a in orders_run.anomalies}
    assert not any(s.issue_id in anomaly_ids for s in orders_run.suggestions)


def test_can_disable(orders, rules_text):
    text = rules_text.replace("anomalies:\n", "anomalies:\n  enabled: false\n")
    assert run_checks(orders[0], text).anomalies == []
