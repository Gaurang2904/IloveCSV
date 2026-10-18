# IloveCSV: Spreadsheet Quality and Validation Engine

Upload an Excel workbook (or CSV) and get back:

1. a **data-quality report** covering column profiles, rule violations and statistical anomalies, with a cell reference for every issue;
2. a **corrected workbook** that contains only the fixes a person approved, with every changed cell highlighted and annotated;
3. an **audit trail** with SHA-256 of the input, rules and output, who approved what and when, and a re-validation of the corrected file.

Built with Python, pandas, openpyxl, scikit-learn, SQLAlchemy (SQLite), FastAPI and Streamlit.

![HTML audit report](reports/figures/report_html.png)

## What it does

| Capability | How |
|---|---|
| **Rule-based validation** | A YAML rule file covers required fields, types, numeric and date ranges, not-in-future dates, allowed categories, regex, length, uniqueness and composite keys. It also supports **cross-column checks** (`ship_date >= order_date`), **conditional requirements** (`when status == 'Shipped'` require `ship_date`) and **lookups** against another sheet. |
| **Data profiling** | Per column: inferred type and how much of the column fits it, missing rate (with configurable null tokens such as `N/A` and `-`), distinct count, top values, quantiles, histogram, date range, text shape patterns, extra spaces, case variants, numbers and dates stored as text, and Excel serial dates. Per sheet: exact duplicate rows and skipped blank rows. |
| **Anomaly detection** | Robust z-score (median/MAD) per numeric column, an IQR fallback, rare categories, and a row-level **Isolation Forest** after PCA whitening, which catches combinations that are each normal on their own. Anomalies are **kept separate from errors**: they never get an automatic fix, and a reviewer confirms or dismisses each one. |
| **Safe correction workflow** | The tool suggests fixes only where the intended value is clear: trimming spaces, normalising category case, fuzzy-matching typos (with a margin over the runner-up), converting text numbers and dates (ambiguous `03/04/2024` gets low confidence), converting Excel serial dates, removing duplicate rows, or applying a fix expression declared in the rules. Missing values are **never imputed** and out-of-range numbers are never guessed. Only **approved** fixes are written. Formulas are never overwritten. |
| **Audit report** | HTML, Excel (Summary / Profile / Issues / Anomalies / Changes / Skipped) and JSON, all built from one data structure. The corrected file is **re-validated with the same rules**, so the report shows before and after counts, including any new problems a fix exposed. |
| **API engineering** | FastAPI with API-key auth (keys stored as SHA-256 hashes), per-user isolation, a streaming body-size limit, extension and magic-byte checks, zip-bomb and macro rejection, RFC 9457 `application/problem+json` errors, and a SQL-backed audit log. |
| **Testing** | Seeded synthetic workbooks with **known injected errors and a ground-truth manifest**. Tests check that each rule catches *exactly* the cells it should, with no misses and no extras, across several seeds. They also check that a clean workbook produces zero issues. |

## Architecture

```
Streamlit UI ──httpx──► FastAPI (/v1) ──► core engine (pure Python, no web dependencies)
                            │             reader → profiler → rules → anomalies → corrections → report
                            └──► SQLite: users, workbooks, runs, issues, suggestions, audit_events
```

* The **core engine** (`src/ilovecsv/`) is used directly by the CLI and the tests.
* The **API** stores issues and decisions in SQLite. Validation is deterministic, with ids derived from the sheet, row, column and rule. Export therefore re-runs the pipeline and overlays the stored decisions, and the database never has to hold workbook contents.
* The **UI** talks only to the API.

| Module | Role |
|---|---|
| `reader.py` | openpyxl/CSV loading. Finds the header row (skipping title rows), spreads merged header cells, keeps Excel row numbers and column letters, marks formula cells, keeps raw values (so `00123` stays text). |
| `profiling.py` | Column and sheet profiles. |
| `rules/schema.py` | Pydantic models for the YAML. Bad regexes, expressions or bounds fail at load time with a location. |
| `rules/expr.py` | **Safe expression evaluator.** It parses with `ast` against a whitelist and evaluates vectorised over pandas. `eval` and `DataFrame.eval` are never called. Nulls are SQL-style *unknown*, which is not a failure. |
| `rules/engine.py` | Turns rules into `Issue`s with cell references. |
| `anomalies.py` | Statistical detectors and Isolation Forest. |
| `corrections.py` | Generates suggestions and writes only approved ones. |
| `report.py`, `explain.py` | HTML/Excel/JSON reports and plain-English explanations. |
| `api/` | FastAPI app, auth, uploads, problem+json errors, routes. |
| `synth.py` | Synthetic workbooks plus the ground-truth manifest. |

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (source .venv/bin/activate elsewhere)
pip install -e ".[ui,dev]"

python scripts/make_demo.py      # builds examples/demo_orders.xlsx + rules, writes reports/demo/
pytest                           # 107 tests
```

### CLI

```bash
ilovecsv profile examples/demo_orders.xlsx
ilovecsv suggest-rules examples/demo_orders.xlsx > my_rules.yaml
ilovecsv check examples/demo_orders.xlsx --rules examples/rules_orders.yaml --out reports/out
ilovecsv check examples/demo_orders.xlsx --rules examples/rules_orders.yaml --out reports/out --approve-min 0.9
```
`check` exits with code 1 when errors remain, so it can gate a data pipeline in CI.

### API + UI

Run each command in its **own terminal**, from the project folder with the venv active. `uvicorn` keeps running, so anything pasted after it in the same terminal will not run.

```bash
# terminal 1 (once): create a key, it is printed only once
python scripts/create_api_key.py alice
# terminal 1: the API on port 8010 (docs at http://127.0.0.1:8010/docs)
uvicorn ilovecsv.api.main:app --port 8010
# terminal 2: the UI (paste the key into the sidebar)
streamlit run app/streamlit_app.py
```

The UI checks the API URL before doing anything else. It tells you if nothing is listening there, or if the port belongs to a different application (port 8000 is often taken by another dev server).

The UI walks through **Upload → Profile → Rules → Issues → Review fixes → Export**. The rule editor and the form builder edit the same YAML. "Use the demo workbook" loads the synthetic file and its rules.

Settings use environment variables: `ILOVECSV_DATA_DIR` (default `./data`), `ILOVECSV_MAX_UPLOAD_MB` (25), `ILOVECSV_MAX_UNCOMPRESSED_MB` (200), `ILOVECSV_DATABASE_URL`.

## Rule file

```yaml
version: 1
dayfirst: true                 # how to read 03/04/2024
null_values: ["", "NA", "N/A", "-"]
anomalies: {exclude: [total]}  # skip derived columns
sheets:
  - sheet: Orders
    key: [order_id]
    checks: {whitespace: true, duplicates: true}
    columns:
      order_id:   {required: true, regex: "^ORD-\\d{5}$"}
      order_date: {required: true, type: date, min: 2020-01-01, not_future: true}
      quantity:   {required: true, type: int, min: 1, max: 1000}
      status:     {required: true, allowed: [Pending, Shipped, Delivered, Cancelled]}
    rules:
      - id: ship_after_order
        when: "ship_date is not null"
        check: "ship_date >= order_date"
      - id: total_matches
        check: "abs(total - quantity * unit_price) <= 0.01"
        fix: {column: total, to: "round(quantity * unit_price, 2)"}
      - id: shipped_needs_date
        when: "status == 'Shipped'"
        require: [ship_date]
      - id: customer_exists
        lookup: {column: customer_id, sheet: Customers, key: customer_id}
```

A single-sheet layout (`sheet`, `columns` and `rules` at the top level) also works. Expressions support column names (use backticks for names with spaces), numbers, strings, `+ - * / %`, comparisons, `in [...]`, `and/or/not`, `is null`, and `abs round len lower upper strip is_null not_null today date days_between coalesce`. Date minus date gives days.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/workbooks` | Upload `.xlsx`/`.csv` (multipart) |
| GET | `/v1/workbooks`, `/v1/workbooks/{id}` | List or describe workbooks |
| GET | `/v1/workbooks/{id}/profile` | Column profiles |
| GET | `/v1/workbooks/{id}/rules/suggested` | Draft rule file from the profile |
| POST | `/v1/rules/validate` | Check a rule file |
| POST | `/v1/workbooks/{id}/runs` | Validate (YAML in the body, or omit it to use suggested rules) |
| GET | `/v1/runs/{id}` | Summary, plus before/after counts once exported |
| GET | `/v1/runs/{id}/issues` | Filter by kind, severity, rule, column, sheet, status and `q`; paginated |
| GET | `/v1/runs/{id}/stats` | SQL `GROUP BY` counts by rule and by column |
| GET | `/v1/runs/{id}/issues/{issue_id}` | Issue, explanation, suggestions, full row |
| PATCH | `/v1/runs/{id}/issues/{issue_id}` | Confirm or dismiss an **anomaly** (409 for rule errors) |
| GET | `/v1/runs/{id}/suggestions` | Suggested corrections |
| POST | `/v1/runs/{id}/decisions` | Approve/reject by id, or bulk-approve by confidence, rule or method |
| POST | `/v1/runs/{id}/export` | Write corrected file and reports, then re-validate |
| GET | `/v1/runs/{id}/report?format=html\|xlsx\|json` | Report (a preview before export) |
| GET | `/v1/runs/{id}/corrected` | Corrected workbook |
| GET | `/v1/runs/{id}/audit-log` | Every recorded action |

Every error is `application/problem+json`:

```json
{"type": "/problems/file-too-large", "title": "Content Too Large", "status": 413,
 "detail": "The upload exceeds the 25 MB limit.", "instance": "/v1/workbooks"}
```

## Results on the demo workbook

The demo has 301 order rows and 40 customer rows. Problems were injected on purpose: 26 rule violations (18 errors and 8 warnings) across 17 rule types, and 5 anomalies.

| | Before | After approving fixes with confidence ≥ 0.85 |
|---|---|---|
| Errors | 18 | 17 |
| Warnings (format problems) | 8 | 1 |
| Quality score (cells without errors) | 99.48% | 99.51% |

* **Rule detection:** every injected error is found, and nothing else fires. The tests check this per rule and on three other seeds.
* **Anomalies:** all 5 are found, including 2 that are only unusual as *combinations* (normal quantity, normal shipping cost, wrong pairing). The clean version of the workbook produces 0 anomalies.
* **Why the error count drops only by one:**
  * Fixing the typo `Shiped` → `Shipped` makes the conditional rule *"Shipped orders need a ship date"* apply to that row, so re-validation reports a **new** error. That is the point of re-validating.
  * The approved duplicate-row deletion was **kept and marked** instead of executed, because the workbook has formulas that row deletion could break.
  * Most remaining errors (missing ids, impossible quantities) need a human. The tool does not guess them.

## Results on real data

I also ran it on the public [BODMAS](https://whyisyoung.github.io/BODMAS/) malware metadata CSV: 134,435 rows of `sha, timestamp, family`, 12 MB. The file itself is not in this repo.

* Run time through the API: upload 0.8 s, profile + validation with suggested rules 9 s, export with re-validation 11 s.
* The suggested rules made `sha` the key with a 64-character length rule. That flagged **7 rows whose id is `VirusShare_<md5>` (43 characters) instead of a SHA-256**.
* The profiler flagged a **dominant value in `timestamp`**: `2020-01-01 00:00:00+00:00` appears in 5,877 rows (4.4%), while a typical timestamp appears once. That pattern looks like a placeholder for unknown dates.
* Real data exposed three problems the synthetic tests had missed, all now fixed:
  * Profiling took 66 s because `strptime` was tried in about 16 formats for every string. It now takes 2.8 s, using shape checks, `fromisoformat` and caching.
  * ISO timestamps with a timezone were not recognised as dates.
  * Re-validating a corrected CSV looked for the wrong sheet name, because a CSV's sheet name comes from its file name.

## Design decisions

* **Errors vs anomalies.** An error means a stated rule was broken, so it is confirmed. An anomaly means the value is statistically unusual, so it is a question for a person. Mixing the two produces either alert fatigue or silent "corrections" of legitimate outliers.
* **No `eval`.** Rule expressions come from users, so they are parsed into an AST and checked node by node against a whitelist. The test-suite tries `__import__`, attribute access, comprehensions, lambdas and the walrus operator.
* **Deterministic ids.** Issue and suggestion ids are hashes of sheet, row, column and rule. The same input and rules always give the same report, which makes runs comparable and lets the API re-derive a run instead of storing workbook contents.
* **Raw values first.** Nothing is coerced on load, so the engine can tell `"12"` (text) from `12`, and `00123` from `123`. For CSV, which has no cell types, plain numbers and ISO dates are treated as canonical.
* **Whitened Isolation Forest.** A plain Isolation Forest missed the correlated-pair anomalies, which scored no worse than the corners of the uniform data. Rotating the robust-scaled data with PCA, fitted on the central 90% of rows, and thresholding the robust z of the isolation scores separates them cleanly: about -9 against at most -3.8 on clean data, across seeds.
* **404, not 403, for other users' resources.** This avoids revealing which ids exist.

## Limitations

* openpyxl does not preserve charts or images when re-saving a workbook. The tool warns when the input contains them.
* Row deletion is skipped in workbooks that contain formulas or merged body cells. The row is marked instead.
* Validation is synchronous: about 10 s for 134k rows × 3 columns. Much larger files would need a background job queue.
* `.xls` and `.xlsm` files are rejected by design.
* `explain.py` uses fixed templates. It is the single place to plug in an LLM later for context-specific explanations.

## Project layout

```
src/ilovecsv/        core engine, API (api/), report template (templates/)
app/                 Streamlit UI
tests/               107 tests: per-rule ground truth, anomalies, corrections, audit, API, CLI
scripts/             make_demo.py, create_api_key.py
examples/            demo workbook + rule file (synthetic)
reports/demo/        demo outputs: corrected workbook, HTML/Excel/JSON audit
```
