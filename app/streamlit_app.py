"""IloveCSV web UI. Everything goes through the FastAPI backend.

    uvicorn ilovecsv.api.main:app            # terminal 1
    streamlit run app/streamlit_app.py       # terminal 2
Set ILOVECSV_API_URL / ILOVECSV_API_KEY, or enter them in the sidebar.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO_XLSX = ROOT / "examples" / "demo_orders.xlsx"
DEMO_RULES = ROOT / "examples" / "rules_orders.yaml"
STEPS = ["1 · Upload", "2 · Profile", "3 · Rules", "4 · Issues", "5 · Review fixes", "6 · Export"]

st.set_page_config(page_title="IloveCSV", page_icon="🧹", layout="wide")
ss = st.session_state
ss.setdefault("nav", STEPS[0])
if "nav_target" in ss:  # set by go(); applied before the radio widget is created
    ss.nav = ss.pop("nav_target")
ss.setdefault("rules_yaml", "")


# ----------------------------------------------------------------------------- API client
def api() -> httpx.Client:
    return httpx.Client(base_url=ss.api_url, headers={"X-API-Key": ss.api_key}, timeout=120)


def call(method: str, path: str, **kw):
    """Call the API; show problem+json errors nicely and return None on failure."""
    try:
        with api() as c:
            r = c.request(method, path, **kw)
    except httpx.HTTPError as e:
        st.error(f"Cannot reach the API at {ss.api_url}: {e}")
        return None
    if r.status_code >= 400:
        try:
            p = r.json()
            st.error(f"**{p.get('title', r.status_code)}**: {p.get('detail', '')}")
            for err in p.get("errors", [])[:10]:
                where = ".".join(err.get("loc", [])) or (f"line {err['line']}" if "line" in err else "")
                st.caption(f"• {where}: {err.get('msg')}")
        except ValueError:
            st.error(f"HTTP {r.status_code}: {r.text[:300]}")
        return None
    return r


def as_text(df: pd.DataFrame) -> pd.DataFrame:
    """Cells mix numbers, text and dates; Arrow (used by st.dataframe) needs one type per column."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v))
    return df


def go(step: str) -> None:
    ss.nav_target = step
    st.rerun()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧹 IloveCSV")
    st.caption("Spreadsheet quality and validation engine")
    ss.api_url = st.text_input("API URL", os.environ.get("ILOVECSV_API_URL", "http://127.0.0.1:8010"))
    ss.api_key = st.text_input("API key", os.environ.get("ILOVECSV_API_KEY", ""), type="password",
                               help="Create one with: python scripts/create_api_key.py yourname")
    st.radio("Steps", STEPS, key="nav", label_visibility="collapsed")
    ss.step = ss.nav
    if ss.get("workbook"):
        st.divider()
        st.caption(f"Workbook: **{ss.workbook['filename']}**")
    if ss.get("run"):
        s = ss.run["summary"]
        st.caption(f"Run `{ss.run['id']}` · {s['errors']} errors · {s['warnings']} warnings · "
                   f"{s['anomalies']} anomalies")

@st.cache_data(ttl=10, show_spinner=False)
def check_api(url: str) -> str | None:
    """Return None if `url` is a live IloveCSV API, else a message saying what is wrong."""
    try:
        r = httpx.get(f"{url.rstrip('/')}/health", timeout=3)
    except httpx.HTTPError:
        return (f"Nothing is answering at {url}. Start the API in its own terminal: "
                "`uvicorn ilovecsv.api.main:app --port 8010`")
    try:
        ok = r.json().get("service") == "ilovecsv"
    except ValueError:
        ok = False
    if not ok:
        return (f"The server at {url} is a different application, not the IloveCSV API (port already in use?). "
                "Start IloveCSV on a free port, e.g. `uvicorn ilovecsv.api.main:app --port 8010`, "
                "and put that URL in the sidebar.")
    return None


if problem := check_api(ss.api_url):
    st.error(problem)
    st.stop()
if not ss.api_key:
    st.info("Enter your API key in the sidebar to start. Create one with `python scripts/create_api_key.py yourname`.")
    st.stop()


def need(key: str, message: str) -> None:
    if not ss.get(key):
        st.warning(message)
        st.stop()


# ----------------------------------------------------------------------------- 1 upload
if ss.step == STEPS[0]:
    st.header("Upload a workbook")
    st.write("Upload an `.xlsx` or `.csv` file. Macro-enabled and legacy `.xls` files are rejected.")
    f = st.file_uploader("Workbook", type=["xlsx", "csv"])
    c1, c2 = st.columns(2)
    if c1.button("Upload", type="primary", disabled=f is None):
        r = call("POST", "/v1/workbooks", files={"file": (f.name, f.getvalue())})
        if r:
            ss.workbook, ss.run = r.json(), None
            ss.rules_yaml = ""
            st.success("Uploaded.")
    if DEMO_XLSX.exists() and c2.button("Use the demo workbook (synthetic orders with known errors)"):
        with open(DEMO_XLSX, "rb") as fh:
            r = call("POST", "/v1/workbooks", files={"file": (DEMO_XLSX.name, fh.read())})
        if r:
            ss.workbook, ss.run = r.json(), None
            ss.rules_yaml = DEMO_RULES.read_text(encoding="utf-8") if DEMO_RULES.exists() else ""
            st.success("Demo workbook uploaded, and its rule file loaded in step 3.")
    if ss.get("workbook"):
        w = ss.workbook
        st.subheader(w["filename"])
        st.caption(f"SHA-256 `{w['sha256']}` · {w['size_bytes']:,} bytes")
        for warn in w["warnings"]:
            st.warning(warn)
        st.dataframe(pd.DataFrame([{"sheet": s["name"], "rows": s["rows"], "columns": len(s["columns"]),
                                    "header row": s["header_row"]} for s in w["sheets"]]), hide_index=True)
        if st.button("Next: profile →"):
            go(STEPS[1])

# ----------------------------------------------------------------------------- 2 profile
elif ss.step == STEPS[1]:
    need("workbook", "Upload a workbook first.")
    st.header("Data profile")
    r = call("GET", f"/v1/workbooks/{ss.workbook['id']}/profile")
    if r:
        for sheet in r.json()["sheets"]:
            st.subheader(f"{sheet['name']}")
            m = st.columns(4)
            m[0].metric("Rows", sheet["n_rows"])
            m[1].metric("Columns", sheet["n_cols"])
            m[2].metric("Duplicate rows", sheet["n_duplicate_rows"])
            m[3].metric("Blank rows skipped", len(sheet["dropped_blank_rows"]))
            table = pd.DataFrame([{
                "column": c["name"], "type": c["inferred_type"], "type share": c["type_share"],
                "missing %": round(100 * c["missing_rate"], 1), "distinct": c["n_distinct"],
                "min": (c.get("numeric") or {}).get("min", (c.get("dates") or {}).get("min")),
                "max": (c.get("numeric") or {}).get("max", (c.get("dates") or {}).get("max")),
                "extra spaces": c["n_whitespace"], "numbers as text": c["n_numeric_text"],
                "dates as text": c["n_date_text"], "id-like": c["is_identifier"],
            } for c in sheet["columns"]])
            st.dataframe(as_text(table), hide_index=True, width="stretch")
            names = [c["name"] for c in sheet["columns"]]
            pick = st.selectbox("Inspect a column", names, key=f"col-{sheet['name']}")
            col = next(c for c in sheet["columns"] if c["name"] == pick)
            a, b = st.columns(2)
            with a:
                st.caption("Top values")
                st.bar_chart(pd.DataFrame(col["top_values"]).set_index("value")["count"]
                             if col["top_values"] else pd.Series(dtype=int))
            with b:
                if col.get("numeric"):
                    h = col["numeric"]["histogram"]
                    labels = [f"{h['edges'][i]:,.4g}–{h['edges'][i + 1]:,.4g}" for i in range(len(h["counts"]))]
                    st.caption("Distribution")
                    st.bar_chart(pd.Series(h["counts"], index=labels))
                if col["case_variants"]:
                    st.caption("Case / spacing variants")
                    st.json(col["case_variants"][:5])
    if st.button("Next: rules →"):
        go(STEPS[2])

# ----------------------------------------------------------------------------- 3 rules
elif ss.step == STEPS[2]:
    need("workbook", "Upload a workbook first.")
    st.header("Validation rules")
    st.write("Rules live in YAML, so they can be reviewed, versioned and audited. "
             "The form below edits the same YAML.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Suggest rules from the data"):
        r = call("GET", f"/v1/workbooks/{ss.workbook['id']}/rules/suggested")
        if r:
            ss.rules_yaml = r.json()["rules_yaml"]
    if DEMO_RULES.exists() and c2.button("Load demo rules"):
        ss.rules_yaml = DEMO_RULES.read_text(encoding="utf-8")
    if c3.button("Check syntax"):
        r = call("POST", "/v1/rules/validate", json={"rules_yaml": ss.rules_yaml})
        if r:
            st.success(f"Valid: {r.json()['sheets']}")

    ss.rules_yaml = st.text_area("Rule file (YAML)", ss.rules_yaml, height=420)

    with st.expander("Add a rule with the form builder"):
        st.caption("Adding through the form re-writes the YAML, so comments are dropped.")
        sheets = {s["name"]: s["columns"] for s in ss.workbook["sheets"]}
        sheet = st.selectbox("Sheet", list(sheets))
        kind = st.radio("Rule kind", ["Column rule", "Cross-column check", "Conditional required"], horizontal=True)
        new_rule: tuple[str, object] | None = None
        if kind == "Column rule":
            col = st.selectbox("Column", sheets[sheet])
            rtype = st.selectbox("Constraint", ["required", "type", "range", "allowed values", "known aliases",
                                                "format (email / phone)", "regex", "unique", "not in the future"])
            spec: dict = {}
            if rtype == "required":
                spec["required"] = True
            elif rtype == "type":
                spec["type"] = st.selectbox("Type", ["int", "float", "date", "bool", "text"])
            elif rtype == "range":
                lo = st.text_input("Minimum (number or YYYY-MM-DD, blank = none)")
                hi = st.text_input("Maximum (number or YYYY-MM-DD, blank = none)")
                for k, v in (("min", lo), ("max", hi)):
                    if v.strip():
                        spec[k] = yaml.safe_load(v)
            elif rtype == "allowed values":
                vals = st.text_input("Allowed values, comma separated")
                spec["allowed"] = [v.strip() for v in vals.split(",") if v.strip()]
            elif rtype == "known aliases":
                st.caption("Spellings that should become a canonical value. Typos that never appear correctly "
                           "elsewhere in the data cannot be found statistically, so list them here.")
                pairs = st.text_input("variant = canonical, comma separated", "Bangalore = Bengaluru, Hydrabad = Hyderabad")
                spec["aliases"] = {k.strip(): v.strip() for k, _, v in
                                   (p.partition("=") for p in pairs.split(",")) if k.strip() and v.strip()}
            elif rtype == "format (email / phone)":
                spec["format"] = st.selectbox("Format", ["email", "phone"])
                if spec["format"] == "phone":
                    spec["phone_digits"] = int(st.number_input("Digits in a local number", 4, 15, 10))
            elif rtype == "regex":
                spec["regex"] = st.text_input("Regular expression (full match)", "^[A-Z]{3}-\\d{5}$")
            elif rtype == "unique":
                spec["unique"] = True
            else:
                spec["not_future"] = True
            spec["severity"] = st.selectbox("Severity", ["error", "warning"])
            new_rule = ("column", (col, spec))
        else:
            rid = st.text_input("Rule id", "my_rule")
            when = st.text_input("Only when (optional)", "")
            if kind == "Cross-column check":
                check = st.text_input("Check (must be true)", "end_date >= start_date")
                body = {"id": rid, "check": check}
            else:
                req = st.multiselect("Required columns", sheets[sheet])
                body = {"id": rid, "require": req}
            if when.strip():
                body["when"] = when
            new_rule = ("table", body)
        if st.button("Add to rule file"):
            doc = yaml.safe_load(ss.rules_yaml or "") or {"version": 1}
            blocks = doc.get("sheets")
            if blocks is None:
                blocks = [{k: doc.pop(k) for k in ("sheet", "key", "checks", "columns", "rules") if k in doc}]
                doc["sheets"] = blocks
            block = next((b for b in blocks if b.get("sheet") in (sheet, None)), None)
            if block is None:
                block = {"sheet": sheet}
                blocks.append(block)
            block.setdefault("sheet", sheet)
            if new_rule[0] == "column":
                col, spec = new_rule[1]
                block.setdefault("columns", {}).setdefault(col, {}).update(spec)
            else:
                block.setdefault("rules", []).append(new_rule[1])
            ss.rules_yaml = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
            st.rerun()

    if st.button("Run validation", type="primary"):
        with st.spinner("Profiling, validating and scoring anomalies…"):
            r = call("POST", f"/v1/workbooks/{ss.workbook['id']}/runs", json={"rules_yaml": ss.rules_yaml or None})
        if r:
            ss.run = r.json()
            go(STEPS[3])

# ----------------------------------------------------------------------------- 4 issues
elif ss.step == STEPS[3]:
    need("run", "Run a validation in step 3 first.")
    rid = ss.run["id"]
    s = call("GET", f"/v1/runs/{rid}").json()["summary"]
    st.header("Issues")
    m = st.columns(5)
    m[0].metric("Quality score", f"{s['quality_score']}%")
    m[1].metric("Errors", s["errors"])
    m[2].metric("Warnings", s["warnings"])
    m[3].metric("Anomalies (open)", s["anomalies_open"])
    m[4].metric("Suggested fixes", s["suggestions"])
    for e in s["rule_errors"]:
        st.warning(f"Rule `{e['rule_id']}` could not run: {e['error']}")

    t_err, t_anom = st.tabs(["Errors and warnings", "Anomalies"])
    with t_err:
        f1, f2 = st.columns(2)
        sev = f1.selectbox("Severity", ["all", "error", "warning"])
        rule = f2.selectbox("Rule", ["all", *s["by_rule"].keys()])
        params = {"kind": "error", "limit": 1000}
        if sev != "all":
            params["severity"] = sev
        if rule != "all":
            params["rule_id"] = rule
        items = call("GET", f"/v1/runs/{rid}/issues", params=params).json()["items"]
        df = pd.DataFrame(items)
        if df.empty:
            st.success("No errors or warnings for this filter.")
        else:
            st.dataframe(as_text(df[["cell_ref", "rule_id", "severity", "value", "message"]]), hide_index=True,
                         width="stretch")
            pick = st.selectbox("Inspect an issue", df["id"], format_func=lambda i: f"{i} · " +
                                df.set_index("id").loc[i, "cell_ref"])
            if pick:
                d = call("GET", f"/v1/runs/{rid}/issues/{pick}").json()
                st.info(d["why"])
                if d["row"]:
                    st.caption("Full row")
                    st.dataframe(as_text(pd.DataFrame([d["row"]])), hide_index=True)
    with t_anom:
        st.caption("Statistically unusual values that break no rule. Confirm the real problems and dismiss the rest. "
                   "Anomalies are never auto-corrected.")
        items = call("GET", f"/v1/runs/{rid}/issues", params={"kind": "anomaly", "limit": 1000}).json()["items"]
        if not items:
            st.success("No anomalies detected.")
        for a in items:
            with st.container(border=True):
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.markdown(f"**{a['cell_ref']}** · `{a['detector']}` · value `{a['value']}` · "
                            f"status **{a['status']}**  \n{a['message']}")
                if c2.button("Confirm", key=f"c-{a['id']}"):
                    call("PATCH", f"/v1/runs/{rid}/issues/{a['id']}", json={"status": "confirmed_error"})
                    st.rerun()
                if c3.button("Dismiss", key=f"d-{a['id']}"):
                    call("PATCH", f"/v1/runs/{rid}/issues/{a['id']}", json={"status": "dismissed"})
                    st.rerun()
    if st.button("Next: review fixes →"):
        go(STEPS[4])

# ----------------------------------------------------------------------------- 5 review
elif ss.step == STEPS[4]:
    need("run", "Run a validation in step 3 first.")
    rid = ss.run["id"]
    st.header("Review suggested corrections")
    st.write("Only the fixes you approve are written to the corrected workbook. "
             "Missing values and out-of-range numbers are never guessed.")
    items = call("GET", f"/v1/runs/{rid}/suggestions").json()["items"]
    if not items:
        st.info("No automatic corrections are available for this run.")
    else:
        df = as_text(pd.DataFrame(items))
        df["confidence"] = df["confidence"].astype(float)
        df["decision"] = df["status"].map({"pending": "pending", "approved": "approve", "rejected": "reject"})
        edited = st.data_editor(
            df[["id", "cell_ref", "old", "new", "method", "confidence", "rule_id", "decision"]],
            column_config={
                "decision": st.column_config.SelectboxColumn("Decision", options=["pending", "approve", "reject"]),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "old": st.column_config.TextColumn("Old"), "new": st.column_config.TextColumn("New"),
            },
            disabled=["id", "cell_ref", "old", "new", "method", "confidence", "rule_id"],
            hide_index=True, width="stretch",
        )
        unsaved = int((edited["decision"] != df["decision"]).sum())
        if unsaved:
            st.warning(f"{unsaved} decision(s) not saved yet. Click **Save decisions** (or Next, which saves them).")

        def save_decisions() -> None:
            approve = edited.loc[edited["decision"] == "approve", "id"].tolist()
            reject = edited.loc[edited["decision"] == "reject", "id"].tolist()
            r = call("POST", f"/v1/runs/{rid}/decisions", json={"approve": approve, "reject": reject})
            if r:
                st.success(f"Saved: {r.json()['suggestions_by_status']}")

        c1, c2 = st.columns(2)
        if c1.button("Save decisions", type="primary"):
            save_decisions()
        with c2:
            thr = st.slider("Bulk approve pending fixes with confidence ≥", 0.5, 1.0, 0.9, 0.01)
            if st.button("Bulk approve"):
                r = call("POST", f"/v1/runs/{rid}/decisions", json={"approve_min_confidence": thr})
                if r:
                    st.success(f"Approved {r.json()['changed']} suggestions.")
                    st.rerun()
    if st.button("Next: export →"):
        if items and unsaved:
            save_decisions()
        go(STEPS[5])

# ----------------------------------------------------------------------------- 6 export
elif ss.step == STEPS[5]:
    need("run", "Run a validation in step 3 first.")
    rid = ss.run["id"]
    st.header("Export the corrected workbook and audit report")
    counts = call("GET", f"/v1/runs/{rid}").json()["summary"]["suggestions_by_status"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Approved fixes", counts["approved"])
    c2.metric("Pending (will NOT be applied)", counts["pending"])
    c3.metric("Rejected", counts["rejected"])
    if counts["approved"] == 0:
        st.warning(f"No fixes are approved, so the corrected workbook will be an unchanged copy of the original "
                   f"({counts['pending']} suggestions are still pending). Approve fixes in step 5 first. "
                   "In the table, set Decision to approve, then click **Save decisions**.")
        if st.button("← Go to Review fixes"):
            go(STEPS[4])
    annotate = st.checkbox("Highlight changed cells and add a comment with the old value", value=True)
    label = "Export" if counts["approved"] else "Export anyway (unchanged copy + report)"
    if st.button(label, type="primary"):
        with st.spinner("Applying approved fixes and re-validating…"):
            r = call("POST", f"/v1/runs/{rid}/export", json={"annotate": annotate})
        if r:
            ss.export = r.json()
    run = call("GET", f"/v1/runs/{rid}").json()
    if run["status"] == "exported":
        before, after = run["summary"], run["after"]
        m = st.columns(3)
        m[0].metric("Quality score", f"{after['quality_score']}%",
                    f"{after['quality_score'] - before['quality_score']:+.2f}")
        m[1].metric("Errors", after["errors"], after["errors"] - before["errors"], delta_color="inverse")
        m[2].metric("Warnings", after["warnings"], after["warnings"] - before["warnings"], delta_color="inverse")
        if ss.get("export") and ss.export.get("skipped"):
            st.warning("Some approved fixes were not applied:")
            st.dataframe(as_text(pd.DataFrame(ss.export["skipped"])[["cell_ref", "method", "reason"]]), hide_index=True)
        stem = Path(ss.workbook["filename"]).stem
        d = st.columns(4)
        corrected = call("GET", f"/v1/runs/{rid}/corrected")
        if corrected:
            suffix = ".csv" if ss.workbook["file_type"] == "csv" else ".xlsx"
            d[0].download_button("Corrected workbook", corrected.content, f"{stem}_corrected{suffix}")
        for i, fmt in enumerate(["html", "xlsx", "json"], start=1):
            rep = call("GET", f"/v1/runs/{rid}/report", params={"format": fmt})
            if rep:
                d[i].download_button(f"Audit report (.{fmt})", rep.content, f"{stem}_audit.{fmt}")
        with st.expander("Audit log"):
            st.dataframe(as_text(pd.DataFrame(call("GET", f"/v1/runs/{rid}/audit-log").json()["items"])),
                         hide_index=True, width="stretch")
