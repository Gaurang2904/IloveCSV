"""Patterns first seen on real data (BODMAS metadata): hash ids, odd ids, placeholder dates."""

import random

from ilovecsv.pipeline import run_checks
from ilovecsv.profiling import profile_workbook
from ilovecsv.reader import read_workbook
from ilovecsv.rules.schema import load_rules
from ilovecsv.rules.suggest import suggest_rules


def _metadata_csv(path, n=500):
    rng = random.Random(0)
    lines = ["sha,timestamp,family"]
    for i in range(n):
        sha = f"{rng.getrandbits(256):064x}"
        ts = f"2019-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{i % 60:02d}:{rng.randint(0, 59):02d}+00:00"
        if i % 10 == 0:
            ts = "2020-01-01 00:00:00+00:00"  # placeholder for unknown dates
        fam = rng.choice(["sfone", "upatre", "wacatac", ""])
        lines.append(f"{sha},{ts},{fam}")
    lines[100] = "VirusShare_" + "a" * 32 + ",2019-05-05 10:00:00+00:00,sfone"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_profile_detects_hash_ids_and_placeholder(tmp_path):
    wb = read_workbook(_metadata_csv(tmp_path / "meta.csv"))
    cols = profile_workbook(wb)["meta"].columns
    assert cols["sha"].is_identifier
    assert cols["sha"].lengths["mode"] == 64 and cols["sha"].lengths["off_mode"] == 1
    assert cols["timestamp"].inferred_type == "date"
    assert cols["timestamp"].dominant_value["value"] == "2020-01-01 00:00:00+00:00"
    assert cols["family"].dominant_value is None  # common categories are not placeholders


def test_suggested_rules_catch_the_odd_id(tmp_path):
    path = _metadata_csv(tmp_path / "meta.csv")
    text = suggest_rules(profile_workbook(read_workbook(path)))
    rules = load_rules(text)
    assert rules.sheets[0].key == ["sha"]
    assert rules.sheets[0].columns["sha"].min_length == 64
    run = run_checks(path, text)
    errors = run.validation.issues
    assert [(i.rule_id, i.cell_ref) for i in errors] == [("length", "meta!A101")]
    assert not any(i.rule_id.startswith("type") for i in errors)  # ISO timestamps are canonical in CSV
