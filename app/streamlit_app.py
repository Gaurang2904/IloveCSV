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
ONE_CLICK = "★ One-click prepare"
PAGES = [ONE_CLICK, *STEPS]  # the numbered steps are the advanced review flow
DEMO_CHURN = ROOT / "examples" / "churn_demo.xlsx"
DATA_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "jsonl", "ndjson", "parquet", "feather", "arrow",
              "arff", "zip", "mat", "npy", "npz", "h5", "hdf5"]

st.set_page_config(page_title="IloveCSV", page_icon="🧹", layout="wide")
ss = st.session_state
ss.setdefault("nav", ONE_CLICK)
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
    st.caption("Dataset quality and preparation for ML")
    ss.api_url = st.text_input("API URL", os.environ.get("ILOVECSV_API_URL", "http://127.0.0.1:8010"))
    ss.api_key = st.text_input("API key", os.environ.get("ILOVECSV_API_KEY", ""), type="password",
                               help="Create one with: python scripts/create_api_key.py yourname")
    st.radio("Steps", PAGES, key="nav", label_visibility="collapsed",
             help="One-click prepare does everything; the numbered steps are the advanced review flow.")
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


# ----------------------------------------------------------------------------- one-click prepare
def _upload(name: str, content: bytes) -> None:
    r = call("POST", "/v1/workbooks", files={"file": (name, content)})
    if r:
        ss.workbook, ss.run, ss.prepared, ss.inspection = r.json(), None, None, None
        ss.rules_yaml = ""


def _inspection_panel(w: dict) -> dict:
    """Structure first: what is inside the file, and which part to prepare. Returns source options."""
    if (ss.get("inspection") or {}).get("workbook_id") != w["id"]:
        r = call("GET", f"/v1/workbooks/{w['id']}/inspect")
        ss.inspection = r.json() if r else {"workbook_id": w["id"], "tables": []}
    insp = ss.inspection
    tables = insp.get("tables", [])
    rows = sum(t["rows"] or 0 for t in tables)
    st.caption(f"**{w['filename']}** · format **{insp.get('format', '?')}** · {len(tables)} table(s) · "
               f"{rows:,} rows · ~{insp.get('memory_estimate_mb', 0)} MB in memory")
    with st.expander("Inspect: tables, columns, documentation", expanded=len(tables) > 1 or bool(insp.get("unsupported"))):
        st.dataframe(pd.DataFrame([{
            "table": t["name"], "rows": f"{t['rows']:,}" + ("" if t["rows_exact"] else " (est.)"),
            "columns": len(t["columns"]), "source": t.get("source", ""), "notes": "; ".join(t.get("notes", []))}
            for t in tables]), hide_index=True, width="stretch")
        if tables:
            pick = st.selectbox("Preview table", [t["name"] for t in tables], key="insp_preview")
            t = next(t for t in tables if t["name"] == pick)
            st.dataframe(as_text(pd.DataFrame(t["sample"])), hide_index=True, width="stretch")
            st.caption("Columns: " + ", ".join(f"`{c['name']}` ({c['kind']})" for c in t["columns"][:40]))
        for d in insp.get("docs", []):
            hints = ", ".join(map(str, d.get("missing_value_hints", [])))
            st.info(f"Documentation **{d.get('name')}** ({d.get('kind')})"
                    + (f": mentions missing-value codes {hints}" if hints else "")
                    + (f"; describes {len(d['columns_mentioned'])} column(s)" if d.get("columns_mentioned") else ""))
        for u in insp.get("unsupported", []):
            st.warning(f"Not tabulated: **{u.get('path') or u.get('name')}**: {u.get('reason')}")
        for warn in insp.get("warnings", []):
            st.warning(warn)
    source: dict = {}
    if len(tables) > 1:
        chosen = st.multiselect("Tables to prepare (default: all)", [t["name"] for t in tables], key="oc_tables")
        if chosen:
            source["sheets" if w["file_type"] == "xlsx" else "tables"] = chosen
    if insp.get("excluded"):
        st.error(f"**{len(insp['excluded'])} malformed record(s) or damaged part(s)** were found. Prepare stops on "
                 "them unless you choose to exclude them.")
        st.dataframe(pd.DataFrame(insp["excluded"]), hide_index=True, width="stretch")
        if st.checkbox(f"Exclude these {len(insp['excluded'])} and import the rest (the output is marked PARTIAL and "
                       "every excluded record is listed in the report)", value=False, key="oc_skip_bad"):
            source["bad_records"] = "skip"
    if any("align_arrays" in w for w in insp.get("warnings", [])):
        if st.checkbox("The same-length arrays above are aligned: combine them into one table", value=False,
                       key="oc_align", help="Element k of each array is one record (e.g. one sample). Equal length "
                                            "alone does not prove this, so it is off unless you know it."):
            source["align_arrays"] = True
    big = not insp.get("full_processing_ok", True)
    if big:
        st.warning(f"Over the full-processing limit ({insp.get('row_limit', 0):,} rows per table). "
                   "Prepare a sample (the output is marked PARTIAL), or pick fewer tables.")
    sample = st.number_input("Process only the first N rows of each table (0 = all)", min_value=0,
                             value=min(100_000, insp.get("row_limit", 0)) if big else 0, step=10_000, key="oc_max_rows")
    if sample:
        source["max_rows"] = int(sample)
    return source


if ss.step == ONE_CLICK:
    st.header("One-click dataset preparation")
    st.write("Upload a dataset: CSV/TSV, Excel, JSON/JSONL, Parquet, Feather, ARFF, ZIP, MATLAB `.mat`, NumPy or "
             "HDF5. IloveCSV shows what is inside, applies only **safe** fixes (formatting, placeholders, "
             "unambiguous conversions), flags everything else for review, re-validates, and exports a cleaned "
             "dataset with a machine-readable report.")
    st.caption("It never imputes, scales, encodes, drops columns or deletes outliers: fit those steps in your ML "
               "pipeline, on the training split only, to avoid data leakage.")
    f = st.file_uploader("Dataset", type=DATA_TYPES, key="oc_file")
    if f is not None and ss.get("oc_uploaded") != (f.name, f.size):
        _upload(f.name, f.getvalue())
        ss.oc_uploaded = (f.name, f.size)
    if DEMO_CHURN.exists() and st.button("Use the demo churn dataset (synthetic, with planted problems)"):
        _upload(DEMO_CHURN.name, DEMO_CHURN.read_bytes())
        ss.oc_uploaded = None
    if ss.get("workbook"):
        w = ss.workbook
        source = _inspection_panel(w)
        chosen = source.get("sheets") or source.get("tables")
        columns = list(dict.fromkeys(c for s in w["sheets"] if not chosen or s["name"] in chosen
                                     for c in s["columns"]))
        guess = next((c for c in columns if str(c).strip().lower() in ("churn", "target", "label", "y", "class")), None)
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("Target column (optional)", ["(none)", *columns],
                              index=(columns.index(guess) + 1) if guess else 0,
                              help="Enables label checks (missing labels, conflicting labels, leakage suspects) "
                                   "and protects the labels from anything but case/spacing fixes.")
        remove_exact = c2.checkbox("Remove exact duplicate rows automatically (off = review them)", value=False,
                                   help="Rows identical in every column, ids included. Leave this off when repeated "
                                        "observations can be legitimate. Near-duplicates are always review-only.")
        parquet = c3.checkbox("Also Parquet", value=False, help="Typed columns for pandas/scikit-learn.")
        labels = ""
        if target != "(none)":
            labels = st.text_input("Valid target labels (optional, comma separated)", placeholder="e.g. No, Yes",
                                   help="Variants such as ' yes ' or 'NO' are normalised to these exact labels; any "
                                        "other value is flagged as unrecognised and never guessed. Empty: use the "
                                        "labels found in the data (placeholders such as '?' are treated as missing).")
        if st.button("Prepare dataset", type="primary"):
            body = {"target": None if target == "(none)" else target, "remove_exact_duplicates": remove_exact,
                    "formats": ["csv", "parquet"] if parquet else ["csv"],
                    "target_labels": [x.strip() for x in labels.split(",") if x.strip()] or None,
                    "source": source or None}
            with st.spinner("Profiling, validating, applying safe fixes, re-validating, exporting…"):
                r = call("POST", f"/v1/workbooks/{w['id']}/prepare", json=body)
            if r:
                ss.prepared = r.json()
                ss.run = {"id": ss.prepared["id"], "summary": ss.prepared["summary"]}
    p = ss.get("prepared")
    if p and ss.get("workbook") and p["workbook_id"] == ss.workbook["id"]:
        rid, after, before = p["id"], p["after"], p["summary"]
        rep = call("GET", f"/v1/runs/{rid}/report", params={"format": "json"})
        report = rep.json() if rep else {}
        if after["errors"] or after["warnings"]:
            st.error(f"**Not fully clean:** {after['errors']} error(s) and {after['warnings']} warning(s) remain. "
                     f"{p['review_required']} item(s) need review.")
        else:
            st.success("**Clean:** the prepared dataset passes every rule.")
        ds = report.get("dataset") or {}
        m = st.columns(5)
        m[0].metric("Rows", " · ".join(f"{ds['rows_before'][s]}→{ds['rows_after'][s]}" for s in ds.get("files", {})))
        m[1].metric("Safe fixes applied", p["safe_fixes_applied"])
        m[2].metric("Needs review", p["review_required"])
        m[3].metric("Unresolved errors", after["errors"])
        m[4].metric("Quality score", f"{after['quality_score']}%", f"{after['quality_score'] - before['quality_score']:+.2f}")
        st.caption(f"Score: `{after['quality_calculation']}`")

        st.subheader("Dataset for ML")
        mls = {e["sheet"]: e for e in report.get("ml", {}).get("sheets", [])}
        for sch in report.get("columns", []):
            roles = [{"role": f"{r} feature", "columns": ", ".join(c)} for r, c in sch["features"].items() if c]
            if sch["excluded"]["target"]:
                roles.append({"role": "target", "columns": sch["excluded"]["target"]})
            roles += [{"role": f"excluded ({k})", "columns": ", ".join(sch["excluded"][k])}
                      for k in ("identifier", "pii", "empty", "constant") if sch["excluded"].get(k)]
            st.markdown(f"**{sch['sheet']}**")
            st.dataframe(pd.DataFrame(roles), hide_index=True, width="stretch")
            ml = mls.get(sch["sheet"], {})
            ts = ml.get("target_summary") or {}
            if ts.get("classes"):
                st.caption(f"Target **{ml['target']}**: {ts['missing']} missing label(s)"
                           + (f", imbalance {ts['imbalance_ratio']}:1" if ts.get("imbalance_ratio") else ""))
                st.bar_chart(pd.DataFrame(ts["classes"]).set_index("label")["count"], horizontal=True)
            for r in sch.get("needs_review", []):
                st.warning(f"**Review before using `{r['column']}`** (not in the feature lists above): "
                           + "; ".join(r["reasons"]) + ". The column is kept in the dataset.")
            if ts.get("unrecognised"):
                st.error("Unrecognised target labels (not guessed): "
                         + ", ".join(f"'{u['label']}' ×{u['count']}" for u in ts["unrecognised"])
                         + f". Valid labels: {', '.join(ts.get('valid_labels') or [])}.")
            if ml.get("missing_target_rows"):
                st.info(f"{len(ml['missing_target_rows'])} row(s) have no usable label (rows "
                        f"{', '.join(map(str, ml['missing_target_rows'][:10]))}). They are kept and flagged.")
            if ml.get("label_conflicts"):
                st.info(f"{len(ml['label_conflicts'])} group(s) of rows have identical features but different labels: "
                        + "; ".join(" / ".join(map(str, g)) for g in ml["label_conflicts"][:5]) + ".")
            if sch["high_missing"]:
                st.caption(f"Over 50% missing (kept; decide in your pipeline): {', '.join(sch['high_missing'])}")

        st.subheader("Downloads")
        stem = Path(ss.workbook["filename"]).stem
        is_xlsx = ss.workbook["file_type"] == "xlsx"
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("**Cleaned file**  \n<small>Your upload with all approved fixes applied (Excel and CSV keep "
                        "their format; other formats come back as CSV tables).</small>", unsafe_allow_html=True)
            d = call("GET", f"/v1/runs/{rid}/corrected")
            if d:
                fname = (d.headers.get("content-disposition", "").split("filename=")[-1].strip('"; ')
                         or f"{stem}_cleaned.{'xlsx' if is_xlsx else 'csv'}")
                label = "Cleaned workbook (.xlsx, changes highlighted)" if is_xlsx else f"Cleaned file ({fname})"
                st.download_button(label, d.content, fname, key="dl-cleaned")
        with a2:
            st.markdown("**ML-ready dataset**  \n<small>One table per sheet: ISO dates, plain numbers, empty cells "
                        "for missing values.</small>", unsafe_allow_html=True)
            for sheet, files in ds.get("files", {}).items():
                for fmt, name in files.items():
                    d = call("GET", f"/v1/runs/{rid}/dataset", params={"format": fmt, "sheet": sheet})
                    if d:
                        st.download_button(f"{name}", d.content, name, key=f"dl-{sheet}-{fmt}")
        with a3:
            st.markdown("**Cleaning report & config**  \n<small>What changed, what needs review, what is "
                        "unresolved.</small>", unsafe_allow_html=True)
            for fmt in ("json", "xlsx", "html"):
                d = call("GET", f"/v1/runs/{rid}/report", params={"format": fmt})
                if d:
                    st.download_button(f"Cleaning report (.{fmt})", d.content, f"{stem}_cleaning_report.{fmt}",
                                       key=f"dl-rep-{fmt}")
            d = call("GET", f"/v1/runs/{rid}/config")
            if d:
                st.download_button("Recipe: prepare config (.yaml)", d.content, f"{stem}_prepare_config.yaml",
                                   key="dl-config")
            for kind, label, fname in (("manifest", "Manifest (.json)", "manifest.json"),
                                       ("feature_schema", "Feature schema (.json)", "feature_schema.json"),
                                       ("script", "Reproduce script (prepare_dataset.py)", "prepare_dataset.py")):
                d = call("GET", f"/v1/runs/{rid}/handoff/{kind}")
                if d:
                    st.download_button(label, d.content, fname, key=f"dl-{kind}")
        ver = (report.get("verification") or {})
        if ver:
            checks = ver.get("checks", [])
            failed = [c for c in checks if not c["passed"]]
            if failed:
                st.error("Post-export verification FAILED: " + "; ".join(f"{c['check']}: {c['detail']}" for c in failed))
            else:
                st.success(f"Post-export verification passed ({len(checks)} checks: outputs reopened, columns and "
                           "row counts match, applied fixes present, re-validated).")
        src = report.get("source") or {}
        if src.get("partial"):
            st.warning(f"**PARTIAL IMPORT:** {src.get('partial_reason') or 'only part of the source was imported'}. "
                       "The results describe the imported part only.")
            if src.get("excluded_records"):
                st.dataframe(pd.DataFrame(src["excluded_records"]), hide_index=True, width="stretch")
        if p["review_required"] and st.button(f"Review the {p['review_required']} item(s) →"):
            go(STEPS[4])

# ----------------------------------------------------------------------------- 1 upload
elif ss.step == STEPS[0]:
    st.header("Upload a workbook")
    st.write("Upload any supported dataset (CSV/TSV, Excel incl. .xls/.xlsm, JSON, Parquet, ZIP, .mat, NumPy, HDF5). "
             "Macros are never run.")
    f = st.file_uploader("Workbook", type=DATA_TYPES)
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
            st.caption("Outcome: resolved (fixed and re-validated) · open (no safe fix; correct the data) · "
                       "pending_review (a fix or anomaly awaits your decision) · approved (applied on export) · "
                       "rejected (fix declined).")
            st.dataframe(as_text(df[["cell_ref", "rule_id", "severity", "value", "message", "resolution",
                                     "resolution_note"]]), hide_index=True, width="stretch")
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
    dups = call("GET", f"/v1/runs/{rid}/duplicates").json()["items"]
    if dups:
        st.subheader(f"Duplicate rows ({len(dups)})")
        st.caption("Removing a row cannot be judged from one cell, so each pair is shown in full. Bulk approval "
                   "never deletes rows unless you tick the box below it.")
        for dp in dups:
            with st.container(border=True):
                how = "exact copy" if dp["exact"] else (
                    f"matches on {', '.join(dp['compared_columns'] or [])}"
                    + (f"; ignoring {', '.join(dp['ignored_columns'])}" if dp["ignored_columns"] else "")
                    + ("; case and spacing ignored" if dp["match"] == "normalized" else ""))
                st.markdown(f"**Row {dp['row']}** repeats **row {dp['duplicate_of']}** ({how}) · "
                            f"decision: **{dp['suggestion_status']}**")
                pair = pd.DataFrame([dp["duplicate_of_values"] or {}, dp["row_values"] or {}],
                                    index=[f"keep · row {dp['duplicate_of']}", f"remove · row {dp['row']}"])
                st.dataframe(as_text(pair), width="stretch")
                if dp["suggestion_id"]:
                    b1, b2, _ = st.columns([1, 1, 4])
                    if b1.button("Remove this row", key=f"dup-a-{dp['issue_id']}"):
                        call("POST", f"/v1/runs/{rid}/decisions", json={"approve": [dp["suggestion_id"]]})
                        st.rerun()
                    if b2.button("Keep both", key=f"dup-r-{dp['issue_id']}"):
                        call("POST", f"/v1/runs/{rid}/decisions", json={"reject": [dp["suggestion_id"]]})
                        st.rerun()
        st.caption("To change what counts as a duplicate, set `checks: {duplicate_columns: [...], "
                   "duplicate_match: exact}` in the rules and validate again.")
        st.subheader("Cell corrections")
    items = call("GET", f"/v1/runs/{rid}/suggestions").json()["items"]
    if not items:
        st.info("No automatic corrections are available for this run.")
    else:
        df = as_text(pd.DataFrame(items))
        df["confidence"] = df["confidence"].astype(float)
        df["decision"] = df["status"].map({"pending": "pending", "approved": "approve", "rejected": "reject"})
        edited = st.data_editor(
            df[["id", "cell_ref", "old", "new", "method", "safety", "confidence", "rule_id", "decision"]],
            column_config={
                "decision": st.column_config.SelectboxColumn("Decision", options=["pending", "approve", "reject"]),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "old": st.column_config.TextColumn("Old"), "new": st.column_config.TextColumn("New"),
            },
            disabled=["id", "cell_ref", "old", "new", "method", "safety", "confidence", "rule_id"],
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
            n_safe = int(((df["safety"] == "safe") & (df["status"] == "pending")).sum())
            if st.button(f"Approve all safe fixes ({n_safe})", disabled=n_safe == 0,
                         help="Lossless formatting and unambiguous conversions only (Safe Auto-Clean)."):
                r = call("POST", f"/v1/runs/{rid}/decisions", json={"approve_safe": True})
                if r:
                    st.success(f"Approved {r.json()['changed']} safe fixes.")
                    st.rerun()
            with st.expander("Advanced: approve by confidence"):
                thr = st.slider("Bulk approve pending fixes with confidence ≥", 0.5, 1.0, 0.9, 0.01)
                with_rows = st.checkbox("Also remove duplicate rows (review the pairs above first)", value=False)
                if st.button("Bulk approve"):
                    r = call("POST", f"/v1/runs/{rid}/decisions",
                             json={"approve_min_confidence": thr, "include_row_deletions": with_rows})
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
    if run["status"] != "exported" and run.get("exported_at"):
        st.info("Decisions changed after the last export, so the previous files are out of date. Export again.")
    if run["status"] == "exported":
        before, after = run["summary"], run["after"]
        if after["errors"] or after["warnings"]:
            st.error(f"**Not fully clean:** {after['errors']} error(s) and {after['warnings']} warning(s) remain "
                     "after cleaning. They are listed below and highlighted orange (with a comment) in the "
                     "corrected workbook.")
        else:
            st.success("**Clean:** the corrected workbook passes every rule in the rule file.")
        m = st.columns(3)
        m[0].metric("Quality score", f"{after['quality_score']}%",
                    f"{after['quality_score'] - before['quality_score']:+.2f}")
        m[1].metric("Unresolved errors", after["errors"], after["errors"] - before["errors"], delta_color="inverse")
        m[2].metric("Unresolved warnings", after["warnings"], after["warnings"] - before["warnings"],
                    delta_color="inverse")
        st.caption(f"Score before: `{before['quality_calculation']}` · after: `{after['quality_calculation']}`. "
                   f"Formula: {before['quality_formula']}.")
        res = before.get("resolutions", {})
        if res:
            st.caption("Issue outcomes: " + " · ".join(f"{k} {v}" for k, v in sorted(res.items())))
        unresolved = call("GET", f"/v1/runs/{rid}/issues", params={"kind": "error", "limit": 1000}).json()["items"]
        unresolved = [u for u in unresolved if u["resolution"] in ("open", "pending_review", "rejected")]
        if unresolved:
            st.subheader("Unresolved after cleaning")
            st.dataframe(as_text(pd.DataFrame(unresolved)[["corrected_cell_ref", "cell_ref", "rule_id", "value",
                                                           "message", "resolution", "resolution_note"]]
                                 .rename(columns={"corrected_cell_ref": "cell (corrected file)",
                                                  "cell_ref": "cell (original)"})),
                         hide_index=True, width="stretch")
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
