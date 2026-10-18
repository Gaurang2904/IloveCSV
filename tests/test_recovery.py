"""Corrupt, malformed and mismatched inputs: a distinct, explicit outcome for each, never silent loss.

Malformed records stop the import unless the user consents (bad_records='skip'); then every excluded
record is listed and the output is marked PARTIAL. Damaged containers are rejected as damaged.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from ilovecsv.adapters import inspect_file
from ilovecsv.cli import main as cli
from ilovecsv.pipeline import prepare
from ilovecsv.reader import (
    ContentMismatchError,
    DamagedFileError,
    MalformedRecordsError,
    UnsupportedFormatError,
    read_workbook,
)
from ilovecsv.sources import SourceOptions

SKIP = SourceOptions(bad_records="skip")


@pytest.fixture
def broken_csv(tmp_path):
    p = tmp_path / "broken.csv"
    p.write_text('id,name,score\n1,Ada,91\n2,"Mina,88\n3,Noah,84,EXTRA\n4,Eli,87\n')
    return p


def test_csv_broken_quote_is_not_silently_swallowed(broken_csv):
    with pytest.raises(MalformedRecordsError) as e:
        read_workbook(broken_csv)
    lines = {p["line"]: p["problem"] for p in e.value.problems}
    assert "never closed" in lines[3] and "4 fields, expected 3" in lines[4]
    assert "Nothing was imported" in str(e.value)
    wb = read_workbook(broken_csv, options=SKIP)
    assert list(wb.first_sheet.df["id"]) == ["1", "4"]  # consented: the two bad records are left out ...
    assert [p["line"] for p in wb.excluded_records] == [3, 4] and wb.partial["excluded_records"] == 2  # ... and listed
    assert wb.excluded_records[0]["raw"] == '2,"Mina,88'


def test_valid_multiline_quoted_field_is_fine(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text('id,note\n1,"two\nlines"\n2,plain\n')
    wb = read_workbook(p)
    assert len(wb.first_sheet.df) == 2 and not wb.excluded_records


def test_rows_of_missing_values_survive_export_and_reload(tmp_path):
    """A one-column table with a NaN row must not lose (and shift) that row anywhere."""
    p = tmp_path / "s.npz"
    np.savez(p, voltage=np.array([4.1, 4.0, np.nan, 4.2, 4.1]), current=np.array([1.0, np.nan, 1.0, 1.0, 0.9]))
    res = prepare(p, tmp_path / "out")
    manifest = json.loads(res.exported.manifest_path.read_text())
    assert all(t["rows_read"] == t["rows_exported"] == 5 for t in manifest["tables"])
    df = pd.read_csv(res.exported.out_dir / res.exported.datasets["voltage"]["csv"])
    assert len(df) == 5 and pd.isna(df["voltage"].iloc[2]) and df["voltage"].iloc[3] == 4.2
    c = tmp_path / "e.csv"
    c.write_text("a,b\n1,2\n,\n3,4\n\n")  # ',' is a record of empty fields; the final empty line is not
    assert len(read_workbook(c).first_sheet.df) == 3


def test_jsonl_bad_and_truncated_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"id": 1, "t": 21.5}\n{"id": 2, "t": oops}\n{"id": 3, "t": 23.1}\n{"id": 4, "t": 24.0')
    with pytest.raises(MalformedRecordsError) as e:
        read_workbook(p)
    probs = {p["line"]: p["problem"] for p in e.value.problems}
    assert probs[2].startswith("not valid JSON") and probs[4].startswith("truncated record")
    wb = read_workbook(p, options=SKIP)
    assert list(wb.first_sheet.df["id"]) == [1, 3] and wb.partial


def test_inspection_reports_instead_of_stopping(broken_csv):
    insp = inspect_file(broken_csv)
    assert [p["line"] for p in insp.excluded] == [3, 4]
    assert any("Prepare stops on them" in w for w in insp.warnings)


def test_partial_import_is_never_called_clean(broken_csv, tmp_path, capsys):
    assert cli(["prepare", str(broken_csv), "--out", str(tmp_path / "a")]) == 2
    err = capsys.readouterr().err
    assert "error [malformed-records]" in err and "Traceback" not in err
    assert cli(["prepare", str(broken_csv), "--skip-bad-records", "--out", str(tmp_path / "b")]) == 1
    out = capsys.readouterr().out
    assert "PARTIAL IMPORT" in out and "; files in" in out and "\n  clean;" not in out
    report = json.loads(next((tmp_path / "b").glob("*_cleaning_report.json")).read_text())
    assert report["source"]["partial_reason"] and len(report["source"]["excluded_records"]) == 2
    assert any(f["finding"] == "excluded on request, not imported" for f in report["findings"])
    cleaned = pd.read_csv(tmp_path / "b" / "broken_cleaned.csv", dtype=str)
    assert list(cleaned["id"]) == ["1", "4"]


def test_json_saved_as_csv_is_a_content_mismatch(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text('[{"id":1,"value":"alpha"},{"id":2,"value":"beta"}]')
    with pytest.raises(ContentMismatchError, match="JSON"):
        read_workbook(p)
    q = tmp_path / "brackets.csv"  # a header that merely starts with '[' is still a CSV
    q.write_text("[id],name\n1,a\n2,b\n")
    assert len(read_workbook(q).first_sheet.df) == 2


def test_damaged_containers(tmp_path):
    good = tmp_path / "good.xlsx"
    pd.DataFrame({"a": range(50)}).to_excel(good, index=False)
    truncated = tmp_path / "t.xlsx"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    with pytest.raises(DamagedFileError):
        read_workbook(truncated)
    npz = tmp_path / "a.npz"
    np.savez(npz, x=np.arange(1000))
    cut = tmp_path / "cut.npz"
    cut.write_bytes(npz.read_bytes()[:300])
    with pytest.raises(DamagedFileError):
        read_workbook(cut)
    junk = tmp_path / "x.bin"
    junk.write_bytes(b"\x00\x01")
    with pytest.raises(UnsupportedFormatError):
        read_workbook(junk)


def _crc_zip(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("good.csv", "id,value\n1,10\n2,20\n")
        z.writestr("damaged.csv", "id,value\n3,30\n4,40\n")
    data = bytearray(path.read_bytes())
    i = data.find(b"3,30")
    data[i] = ord("9")  # change the stored bytes without updating the CRC
    path.write_bytes(bytes(data))


def test_zip_member_crc_failure(tmp_path):
    p = tmp_path / "crc.zip"
    _crc_zip(p)
    with pytest.raises(DamagedFileError) as e:
        read_workbook(p)
    assert "damaged.csv" in str(e.value) and "Intact members: good.csv" in str(e.value)
    wb = read_workbook(p, options=SKIP)
    assert set(wb.sheets) == {"good"} and wb.excluded_records[0]["member"] == "damaged.csv" and wb.partial


def test_flattening_collision_keeps_both_values(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": 1, "a.b": "literal", "a": {"b": "nested"}}\n{"id": 2, "a.b": "x", "a": {"b": "y"}}\n')
    wb = read_workbook(p)
    df = wb.first_sheet.df
    assert list(df["a.b"]) == ["literal", "x"] and list(df["a/b"]) == ["nested", "y"]
    assert any("collision" in w for w in wb.warnings)


def test_date_text_fix_keeps_the_time(tmp_path):
    p = tmp_path / "t.xlsx"
    times = [f"2025/08/{d:02d} 10:{d:02d}" for d in range(1, 21)]
    pd.DataFrame({"reading_time": times, "v": range(20)}).to_excel(p, index=False)
    ex = prepare(p, tmp_path / "out").exported
    out = pd.read_csv(ex.out_dir / "t_cleaned.csv", dtype=str)
    assert out["reading_time"].iloc[0].startswith("2025-08-01") and "10:01" in out["reading_time"].iloc[0]


def test_time_series_gap_and_repeated_reading(tmp_path):
    p = tmp_path / "ts.csv"
    rows = ["device_id,timestamp,temp,volt"]
    for dev in ("A", "B"):
        for m in range(8):
            if dev == "A" and m == 4:
                continue  # a missing reading: a 2-minute gap
            temp, volt = (21.0, 3.6) if dev == "B" and m in (5, 6) else (20 + m * 0.1, 3.5 + m * 0.01)
            rows.append(f"{dev},2026-01-01T10:{m:02d}:00Z,{temp},{volt}")
    p.write_text("\n".join(rows) + "\n")
    res = prepare(p, tmp_path / "out")
    found = {(i.rule_id, i.xl_row) for i in res.run.issues if i.rule_id.startswith("timeseries")}
    assert ("timeseries.gap", 6) in found  # row of A 10:05, after the missing 10:04
    assert ("timeseries.repeated_reading", 15) in found  # B 10:06 repeats B 10:05
    assert len(found) == 2
    assert pd.read_csv(res.exported.out_dir / "ts_cleaned.csv")["timestamp"].is_monotonic_increasing is False  # not sorted


def test_irregular_event_log_gets_no_gap_warnings(tmp_path):
    p = tmp_path / "ev.csv"
    minutes = [0, 7, 9, 31, 40, 95, 101, 160]  # events of varying length: no typical step
    p.write_text("unit_id,event_time,v\n" + "".join(f"U1,2026-01-01T{m // 60:02d}:{m % 60:02d}:00,{i}\n"
                                                    for i, m in enumerate(minutes)))
    res = prepare(p, tmp_path / "out")
    assert res.run.rules.sheets[0].roles  # detected as a series (times rise) ...
    assert not [i for i in res.run.issues if i.rule_id == "timeseries.gap"]  # ... but not regularly sampled


def test_random_dates_do_not_make_a_time_series(tmp_path):
    p = tmp_path / "c.csv"
    days = [5, 2, 9, 1, 7, 3, 8, 4, 6, 10]
    p.write_text("id,signup_date\n" + "".join(f"{i},2024-01-{d:02d}\n" for i, d in enumerate(days)))
    res = prepare(p, tmp_path / "out")
    assert not res.run.rules.sheets[0].roles
    assert not [i for i in res.run.issues if i.rule_id.startswith("timeseries")]


def test_constant_column_and_no_target_needed(tmp_path):
    p = tmp_path / "a_rather_long_unsupervised_file_name.csv"
    p.write_text("key,spend,src\n" + "".join(f"K{i},{100 + i},import_v1\n" for i in range(30)))
    res = prepare(p, tmp_path / "out")
    report = json.loads(res.report.read_text())
    sch = report["columns"][0]
    assert sch["excluded"]["constant"] == ["src"] and "src" not in sum(sch["features"].values(), [])
    assert sch["sheet"] == "a_rather_long_unsupervised_file_name"  # not cut to Excel's 31 characters
    assert not [i for i in res.run.issues if i.rule_id.startswith("target")]
    assert report["ml"]["sheets"][0].get("target") is None


def test_api_statuses_are_distinct(tmp_path):
    from fastapi.testclient import TestClient

    from ilovecsv.api.deps import create_user
    from ilovecsv.api.main import create_app
    from ilovecsv.api.settings import Settings

    app = create_app(Settings(data_dir=tmp_path / "data"))
    with app.state.sessionmaker() as db:
        h = {"X-API-Key": create_user(db, "t")}
    client = TestClient(app, raise_server_exceptions=False)

    def up(name, content):
        return client.post("/v1/workbooks", headers=h, files={"file": (name, content)})

    assert up("x.bin", b"\x00\x01").json()["type"].endswith("unsupported-media-type")
    assert up("x.csv", b'[{"id": 1}]').json()["type"].endswith("content-mismatch")
    assert up("x.npz", b"PK\x03\x04 truncated").json()["type"].endswith("damaged-file")
    r = up("b.csv", b'id,name,score\n1,Ada,91\n2,"Mina,88\n4,Eli,87\n')
    assert r.status_code == 201  # uploading and inspecting is fine ...
    wid = r.json()["id"]
    bad = client.post(f"/v1/workbooks/{wid}/prepare", headers=h, json={})
    assert bad.status_code == 422 and bad.json()["type"].endswith("malformed-records")  # ... preparing needs consent
    assert bad.json()["errors"][0]["line"] == 3
    ok = client.post(f"/v1/workbooks/{wid}/prepare", headers=h, json={"source": {"bad_records": "skip"}})
    assert ok.status_code == 201, ok.text
