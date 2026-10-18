"""API: auth, upload limits, structured errors and the full review workflow."""

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from ilovecsv.api.deps import create_user
from ilovecsv.api.main import create_app
from ilovecsv.api.settings import Settings

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture()
def api(tmp_path):
    app = create_app(Settings(data_dir=tmp_path / "data", max_upload_mb=1.0))
    with app.state.sessionmaker() as db:
        alice = create_user(db, "alice")
        bob = create_user(db, "bob")
    client = TestClient(app, raise_server_exceptions=False)
    return client, {"X-API-Key": alice}, {"X-API-Key": bob}


def upload(client, headers, path, name=None, ctype=XLSX):
    with open(path, "rb") as f:
        return client.post("/v1/workbooks", headers=headers, files={"file": (name or path.name, f, ctype)})


def assert_problem(r, status, slug=None):
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert {"type", "title", "status", "detail", "instance"} <= body.keys()
    assert body["status"] == status
    if slug:
        assert body["type"] == f"/problems/{slug}"
    return body


def test_health_needs_no_key(api):
    client, *_ = api
    assert client.get("/health").json()["status"] == "ok"


def test_missing_and_invalid_key(api, orders):
    client, *_ = api
    r = upload(client, {}, orders[0])
    assert_problem(r, 401, "missing-api-key")
    assert r.headers["www-authenticate"] == "ApiKey"
    assert_problem(client.get("/v1/workbooks", headers={"X-API-Key": "ilc_wrong"}), 401, "invalid-api-key")


def test_other_users_cannot_see_workbook(api, orders):
    client, alice, bob = api
    wid = upload(client, alice, orders[0]).json()["id"]
    assert_problem(client.get(f"/v1/workbooks/{wid}", headers=bob), 404, "not-found")
    assert client.get("/v1/workbooks", headers=bob).json()["items"] == []
    run = client.post(f"/v1/workbooks/{wid}/runs", headers=alice, json={}).json()
    assert_problem(client.get(f"/v1/runs/{run['id']}/issues", headers=bob), 404)


def test_too_large(api, tmp_path):
    client, alice, _ = api
    big = tmp_path / "big.csv"
    big.write_bytes(b"a,b\n" + b"1,2\n" * 400_000)  # ~1.6 MB > 1 MB limit
    body = assert_problem(upload(client, alice, big, ctype="text/csv"), 413, "file-too-large")
    assert "MB" in body["detail"]


def test_wrong_types(api, tmp_path):
    client, alice, _ = api
    doc = tmp_path / "notes.docx"
    doc.write_text("hello")
    assert_problem(upload(client, alice, doc, ctype="text/plain"), 415, "unsupported-media-type")
    fake = tmp_path / "renamed.xlsx"
    fake.write_text("this is not a workbook")
    assert_problem(upload(client, alice, fake), 415, "content-mismatch")
    xlsm = tmp_path / "macro.xlsm"
    xlsm.write_bytes(b"PK\x03\x04")
    assert_problem(upload(client, alice, xlsm), 415, "content-mismatch")  # not a readable zip
    parquet = tmp_path / "fake.parquet"
    parquet.write_text("a,b\n1,2\n")
    assert_problem(upload(client, alice, parquet), 415, "content-mismatch")
    z = tmp_path / "zip.xlsx"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("hello.txt", "hi")
    assert_problem(upload(client, alice, z), 415, "content-mismatch")


def test_macro_workbooks(api, tmp_path):
    """.xlsm is accepted (macros are never run); macros hidden in an .xlsx are refused."""
    from openpyxl import Workbook as XL

    client, alice, _ = api
    good = tmp_path / "plain.xlsx"
    wb = XL()
    wb.active.append(["id", "score"])
    wb.active.append([1, 2.5])
    wb.save(good)
    disguised = tmp_path / "disguised.xlsx"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(disguised, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.writestr("xl/vbaProject.bin", b"fake macro project")
    assert_problem(upload(client, alice, disguised), 415, "unsupported-media-type")
    xlsm = tmp_path / "macro.xlsm"
    xlsm.write_bytes(disguised.read_bytes())
    r = upload(client, alice, xlsm)
    assert r.status_code == 201, r.text
    assert any("macro" in w.lower() for w in r.json()["warnings"])


def test_upload_other_formats_and_inspect(api, tmp_path):
    import json

    client, alice, _ = api
    data = tmp_path / "orders.json"
    data.write_text(json.dumps({"meta": {"n": 2}, "items": [
        {"id": 1, "customer": {"name": "Ann", "city": "Pune"}, "amount": "10.5"},
        {"id": 2, "customer": {"name": "Raj", "city": "Mumbai"}, "amount": "N/A"}]}))
    r = upload(client, alice, data, ctype="application/json")
    assert r.status_code == 201, r.text
    wid = r.json()["id"]
    assert r.json()["file_type"] == "frames"
    insp = client.get(f"/v1/workbooks/{wid}/inspect", headers=alice).json()
    assert insp["format"] == "json"
    cols = [c["name"] for c in insp["tables"][0]["columns"]]
    assert "customer.city" in cols and insp["tables"][0]["rows"] == 2
    prep = client.post(f"/v1/workbooks/{wid}/prepare", headers=alice, json={"source": {"max_rows": 10}})
    assert prep.status_code == 201, prep.text
    assert prep.json()["datasets"]


def test_zip_bomb_guard(api, tmp_path):
    client, alice, _ = api
    bomb = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<x/>")
        zf.writestr("xl/workbook.xml", "<x/>")
        zf.writestr("xl/worksheets/sheet1.xml", b"0" * (300 * 1024 * 1024))
    assert bomb.stat().st_size < 1024 * 1024
    assert_problem(upload(client, alice, bomb), 422, "unsafe-archive")


def test_bad_rules_are_422_with_locations(api, orders):
    client, alice, _ = api
    wid = upload(client, alice, orders[0]).json()["id"]
    r = client.post(f"/v1/workbooks/{wid}/runs", headers=alice, json={"rules_yaml": "columns:\n  a: {required: true\n"})
    body = assert_problem(r, 422, "invalid-rules")
    assert body["errors"][0]["line"] >= 1
    r = client.post("/v1/rules/validate", headers=alice, json={"rules_yaml": "columns:\n  a: {regex: '('}\n"})
    assert_problem(r, 422, "invalid-rules")
    r = client.post(f"/v1/workbooks/{wid}/runs", headers=alice, json={"rules_yaml": 5})
    body = assert_problem(r, 422, "request-validation")
    assert body["errors"]


def test_unknown_route_is_problem(api):
    client, alice, _ = api
    assert_problem(client.get("/v1/runs/nope", headers=alice), 404)
    assert_problem(client.get("/v1/nothing", headers=alice), 404)


def test_full_workflow(api, orders, rules_text):
    client, alice, _ = api
    wb = upload(client, alice, orders[0])
    assert wb.status_code == 201
    wb = wb.json()
    assert [s["name"] for s in wb["sheets"]] == ["Orders", "Customers", "Summary"]

    prof = client.get(f"/v1/workbooks/{wb['id']}/profile", headers=alice).json()
    assert prof["sheets"][0]["n_rows"] == 301
    draft = client.get(f"/v1/workbooks/{wb['id']}/rules/suggested", headers=alice).json()["rules_yaml"]
    assert "columns:" in draft

    run = client.post(f"/v1/workbooks/{wb['id']}/runs", headers=alice, json={"rules_yaml": rules_text})
    assert run.status_code == 201
    run = run.json()
    rid, summary = run["id"], run["summary"]
    assert summary["errors"] > 0 and summary["anomalies"] > 0

    errs = client.get(f"/v1/runs/{rid}/issues", headers=alice, params={"kind": "error", "severity": "error"}).json()
    assert errs["total"] == summary["errors"]
    page = client.get(f"/v1/runs/{rid}/issues", headers=alice, params={"limit": 5, "offset": 5}).json()
    assert len(page["items"]) == 5 and page["total"] == summary["errors"] + summary["warnings"] + summary["anomalies"]
    stats = client.get(f"/v1/runs/{rid}/stats", headers=alice).json()
    assert sum(r["n"] for r in stats["by_rule"]) == page["total"]

    one = client.get(f"/v1/runs/{rid}/issues/{errs['items'][0]['id']}", headers=alice).json()
    assert one["why"] and one["row"]

    # Errors cannot be dismissed; anomalies can be confirmed or dismissed.
    assert_problem(client.patch(f"/v1/runs/{rid}/issues/{errs['items'][0]['id']}", headers=alice,
                                json={"status": "dismissed"}), 409, "not-an-anomaly")
    anomalies = client.get(f"/v1/runs/{rid}/issues", headers=alice, params={"kind": "anomaly"}).json()["items"]
    r = client.patch(f"/v1/runs/{rid}/issues/{anomalies[0]['id']}", headers=alice,
                     json={"status": "confirmed_error", "note": "typo in price"})
    assert r.json()["status"] == "confirmed_error" and r.json()["decided_by"] == "alice"
    assert client.get(f"/v1/runs/{rid}", headers=alice).json()["summary"]["anomalies_confirmed"] == 1

    sugs = client.get(f"/v1/runs/{rid}/suggestions", headers=alice).json()["items"]
    assert sugs and all(s["status"] == "pending" for s in sugs)
    assert_problem(client.post(f"/v1/runs/{rid}/decisions", headers=alice, json={"approve": ["s-nope"]}), 422)
    low = [s["id"] for s in sugs if s["confidence"] < 0.9]
    d = client.post(f"/v1/runs/{rid}/decisions", headers=alice,
                    json={"reject": low, "approve_min_confidence": 0.9}).json()
    # Bulk approval never deletes rows: duplicates are approved one by one after the preview.
    bulk = [s for s in sugs if s["confidence"] >= 0.9 and s["action"] != "delete_row"]
    assert d["suggestions_by_status"]["approved"] == len(bulk)
    dups = client.get(f"/v1/runs/{rid}/duplicates", headers=alice).json()["items"]
    assert len(dups) == 1 and dups[0]["exact"] and dups[0]["row_values"] == dups[0]["duplicate_of_values"]
    assert dups[0]["suggestion_status"] == "pending"
    d = client.post(f"/v1/runs/{rid}/decisions", headers=alice, json={"approve": [dups[0]["suggestion_id"]]}).json()
    assert d["suggestions_by_status"]["approved"] == len(bulk) + 1
    pre = client.get(f"/v1/runs/{rid}/issues", headers=alice, params={"resolution": "approved"}).json()
    assert pre["total"] >= len(bulk)

    assert_problem(client.get(f"/v1/runs/{rid}/corrected", headers=alice), 409, "not-exported")
    preview = client.get(f"/v1/runs/{rid}/report", headers=alice, params={"format": "json"})
    assert preview.json()["output"] is None

    ex = client.post(f"/v1/runs/{rid}/export", headers=alice, json={"annotate": True})
    assert ex.status_code == 200, ex.text
    ex = ex.json()
    assert ex["status"] == "exported"
    assert len(ex["applied"]) + len(ex["skipped"]) == d["suggestions_by_status"]["approved"]
    assert ex["after"]["warnings"] < summary["warnings"]
    assert "Not fully clean" in ex["notice"]
    res = client.get(f"/v1/runs/{rid}", headers=alice).json()["summary"]["resolutions"]
    assert res.get("resolved", 0) > 0 and res.get("open", 0) > 0 and "approved" not in res
    unresolved = client.get(f"/v1/runs/{rid}/issues", headers=alice,
                            params={"resolution": "open", "kind": "error"}).json()["items"]
    assert unresolved and all(i["resolution_note"] for i in unresolved)
    assert any(i["introduced_by_fix"] for i in unresolved)  # Shiped -> Shipped exposes a missing ship date

    f = client.get(f"/v1/runs/{rid}/corrected", headers=alice)
    assert f.status_code == 200 and f.content[:2] == b"PK"
    from openpyxl import load_workbook

    load_workbook(io.BytesIO(f.content))
    html = client.get(f"/v1/runs/{rid}/report", headers=alice)
    assert html.headers["content-type"].startswith("text/html") and b"Change log" in html.content
    audit = client.get(f"/v1/runs/{rid}/report", headers=alice, params={"format": "json"}).json()
    import hashlib

    assert audit["output"]["sha256"] == hashlib.sha256(f.content).hexdigest()
    assert any(a["status"] == "confirmed_error" for a in audit["anomalies"])

    log = client.get(f"/v1/runs/{rid}/audit-log", headers=alice).json()["items"]
    actions = [e["action"] for e in log]
    for expected in ("workbook.uploaded", "run.created", "issue.status_changed", "suggestions.decided",
                     "run.exported", "corrected.downloaded"):
        assert expected in actions
    assert all(e["user"] == "alice" for e in log)


def test_run_without_rules_uses_suggested(api, clean_orders):
    client, alice, _ = api
    wid = upload(client, alice, clean_orders).json()["id"]
    run = client.post(f"/v1/workbooks/{wid}/runs", headers=alice, json={}).json()
    assert run["summary"]["errors"] == 0
    rules = client.get(f"/v1/runs/{run['id']}/rules", headers=alice).json()
    assert rules["rules_yaml"].startswith("# Draft rules")
