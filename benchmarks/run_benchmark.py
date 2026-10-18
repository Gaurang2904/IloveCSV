"""Benchmark: one clean synthetic dataset, known problems planted, written in every tabular format.

For each format, `prepare` runs once (Safe Auto-Clean, no manual decisions) and is scored against the
ground truth:

  detection   precision / recall over planted problems (a planted cell counts as found when an
              error or warning is raised on that row and column; duplicates by row)
  repair      planted problems with a safe, unambiguous fix: is the output value the clean value?
  untouched   cells with no planted problem: are they unchanged in the output? (must be 100%)
  rows        valid rows preserved (nothing is deleted without a rule)
  reproduce   re-running with the saved recipe gives byte-identical datasets
  review      items left for a person (pending suggestions + open anomalies)
  time        seconds for prepare

    python benchmarks/run_benchmark.py [--rows 2000] [--out reports/benchmark]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ilovecsv.pipeline import prepare  # noqa: E402

CITIES = ["Pune", "Mumbai", "Delhi", "Chennai", "Kolkata"]
PLANS = ["basic", "standard", "premium"]


@dataclass
class Planted:
    row: int  # 0-based data row
    column: str | None  # None: the whole row (duplicates)
    kind: str
    dirty: object
    clean: object  # expected output value when the fix is safe; None = must stay flagged, not fixed
    fixable: bool


def clean_frame(n: int, rng: random.Random) -> pd.DataFrame:
    start = date(2023, 1, 1)
    rows = []
    for i in range(n):
        age = rng.randint(18, 80)
        rows.append({
            "customer_id": f"C{10000 + i}",
            "city": rng.choice(CITIES),
            "plan": rng.choice(PLANS),
            "age": age,
            "monthly_spend": round(rng.uniform(5, 300), 2),
            "signup_date": (start + timedelta(days=rng.randint(0, 600))).isoformat(),
            "email": f"user{i}@example.com",
            "churn": rng.choice(["Yes", "No"]),
        })
    return pd.DataFrame(rows)


def plant(df: pd.DataFrame, rng: random.Random, share: float = 0.02) -> tuple[pd.DataFrame, list[Planted]]:
    """Plant each problem type in ~share of the rows; at most one problem per cell."""
    dirty = df.astype(object).copy()
    planted: list[Planted] = []
    used: set[tuple[int, str]] = set()
    n = len(df)
    k = max(3, int(n * share))

    def pick(column: str) -> list[int]:
        rows = [r for r in rng.sample(range(n), min(n, k * 3)) if (r, column) not in used][:k]
        used.update((r, column) for r in rows)
        return rows

    for r in pick("city"):  # extra whitespace: safe trim
        v = df.at[r, "city"]
        dirty.at[r, "city"] = f"  {v} "
        planted.append(Planted(r, "city", "whitespace", dirty.at[r, "city"], v, True))
    for r in pick("plan"):  # case variant of a known category: safe canonicalisation
        v = df.at[r, "plan"]
        dirty.at[r, "plan"] = v.upper()
        planted.append(Planted(r, "plan", "case_variant", v.upper(), v, True))
    for r in pick("monthly_spend"):  # placeholder for missing: safe -> empty cell
        token = rng.choice(["N/A", "?", "null", "-"])
        dirty.at[r, "monthly_spend"] = token
        planted.append(Planted(r, "monthly_spend", "placeholder", token, "", True))
    for r in pick("age"):  # impossible value: flagged, never "fixed"
        bad = rng.choice([-5, 0, 230, 999])
        dirty.at[r, "age"] = bad
        planted.append(Planted(r, "age", "out_of_range", bad, None, False))
    for r in pick("email"):  # malformed email: flagged
        bad = df.at[r, "email"].replace("@", "@@")
        dirty.at[r, "email"] = bad
        planted.append(Planted(r, "email", "invalid_email", bad, None, False))
    for r in pick("churn"):  # missing / unrecognised label: flagged
        bad = rng.choice(["", "Maybe"])
        dirty.at[r, "churn"] = bad or None
        planted.append(Planted(r, "churn", "bad_label", bad, None, False))
    for r in pick("signup_date")[: max(1, k // 2)]:  # unparseable date text: flagged
        dirty.at[r, "signup_date"] = "sometime in March"
        planted.append(Planted(r, "signup_date", "invalid_date", "sometime in March", None, False))
    # Exact duplicates: copies of clean rows appended at the end (flagged; removal is review by default).
    sources = rng.sample([r for r in range(n) if not any(p.row == r for p in planted)], max(2, k // 2))
    dups = dirty.iloc[sources].copy()
    dirty = pd.concat([dirty, dups], ignore_index=True)
    for i, _src in enumerate(sources):
        planted.append(Planted(n + i, None, "exact_duplicate", None, None, False))
    return dirty, planted


# ----------------------------------------------------------------------------- formats
def _as_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columnar formats need one type per column: mixed columns are stored as text (as pandas would)."""
    out = df.copy()
    for c in out.columns:
        kinds = {"text" if isinstance(v, str) else "number" for v in out[c] if v is not None}
        if len(kinds) > 1:
            out[c] = [None if v is None else str(v) for v in out[c]]
    return out


def write_formats(df: pd.DataFrame, d: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    files["csv"] = d / "customers.csv"
    df.to_csv(files["csv"], index=False)
    files["tsv"] = d / "customers.tsv"
    df.to_csv(files["tsv"], index=False, sep="\t")
    files["xlsx"] = d / "customers.xlsx"
    df.to_excel(files["xlsx"], index=False, sheet_name="customers")
    files["json"] = d / "customers.json"
    files["json"].write_text(json.dumps({"exported": "2024-06-01", "records": df.to_dict("records")}, default=str))
    files["jsonl"] = d / "customers.jsonl"
    files["jsonl"].write_text("\n".join(json.dumps(r, default=str) for r in df.to_dict("records")))
    files["parquet"] = d / "customers.parquet"
    _as_text_columns(df).to_parquet(files["parquet"], index=False)
    files["zip"] = d / "customers_bundle.zip"
    with zipfile.ZipFile(files["zip"], "w", zipfile.ZIP_DEFLATED) as z:
        z.write(files["csv"], "data/customers.csv")
        z.writestr("README.txt", "Customer extract. churn is the label (Yes/No).\n")
    return files


# ----------------------------------------------------------------------------- scoring
def _norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v)
    try:
        f = float(s)
        return repr(round(f, 6)) if s.strip() not in ("",) else ""
    except ValueError:
        return s


REPRESENTATION = ("type.stored_as_text", "type.date_as_text")


def score(fmt: str, path: Path, planted: list[Planted], clean: pd.DataFrame, dirty: pd.DataFrame, out: Path,
          config: Path | None = None) -> dict:
    t0 = time.perf_counter()
    src = {"json_records": "records"} if fmt == "json" else None
    res = prepare(path, out, target="churn", target_labels=["No", "Yes"], source=src, config=config)
    elapsed = time.perf_counter() - t0
    run, ex = res.run, res.exported
    table = next(iter(ex.datasets))
    cleaned = pd.read_csv(ex.out_dir / ex.datasets[table]["csv"], dtype=str, keep_default_na=False)

    # A valid value held in the wrong representation for its format (an ISO date typed as text in
    # Excel, a number in a text column of a Parquet file) is a real, safely fixed finding, but not a
    # planted problem: counted separately, not as a false positive.
    representation = [i for i in run.issues if i.rule_id in REPRESENTATION]
    flagged_cells = {(i.xl_row - 2, i.column) for i in run.issues
                     if i.kind == "error" and i.severity in ("error", "warning") and i.xl_row and i.column
                     and i.rule_id not in REPRESENTATION}
    flagged_rows = {i.xl_row - 2 for i in run.issues if i.kind == "error" and i.rule_id == "row.duplicate"
                    and i.xl_row}
    planted_cells = {(p.row, p.column) for p in planted if p.column}
    found = [p for p in planted if ((p.row, p.column) in flagged_cells if p.column else p.row in flagged_rows)]
    # False positives: flags on cells that hold no planted problem (rows of duplicates excluded).
    dup_rows = {p.row for p in planted if p.kind == "exact_duplicate"}
    fp = {c for c in flagged_cells if c not in planted_cells and c[0] not in dup_rows}
    fp_rows = {r for r in flagged_rows if r not in dup_rows}
    tp = len(found)
    precision = tp / (tp + len(fp) + len(fp_rows)) if tp else 0.0
    recall = tp / len(planted)

    fixable = [p for p in planted if p.fixable]
    repaired = [p for p in fixable if _norm(cleaned.at[p.row, p.column]) == _norm(p.clean)]
    # Cells without a planted problem must come out with the same value.
    n_rows = len(clean)
    untouched_total = untouched_same = 0
    for col in clean.columns:
        for r in range(n_rows):
            if (r, col) in planted_cells:
                continue
            untouched_total += 1
            untouched_same += _norm(cleaned.at[r, col]) == _norm(clean.at[r, col])
    rows_preserved = min(len(cleaned), n_rows) / n_rows

    # Reproducibility: the saved recipe applied again gives identical datasets.
    again = prepare(path, out.parent / f"{out.name}_rerun", config=ex.config_path)
    h1 = hashlib.sha256((ex.out_dir / ex.datasets[table]["csv"]).read_bytes()).hexdigest()
    t2 = next(iter(again.exported.datasets))
    h2 = hashlib.sha256((again.exported.out_dir / again.exported.datasets[t2]["csv"]).read_bytes()).hexdigest()

    summary = run.summary()
    by_kind: dict[str, list[int]] = {}
    for p in planted:
        by_kind.setdefault(p.kind, [0, 0])[1] += 1
    for p in found:
        by_kind[p.kind][0] += 1
    return {
        "format": fmt, "rows": len(dirty), "planted": len(planted), "found": tp,
        "false_positive_cells": len(fp) + len(fp_rows), "precision": round(precision, 4), "recall": round(recall, 4),
        "fixable": len(fixable), "repaired_correctly": len(repaired),
        "repair_accuracy": round(len(repaired) / len(fixable), 4) if fixable else None,
        "untouched_cells_unchanged": round(untouched_same / untouched_total, 6),
        "valid_rows_preserved": round(rows_preserved, 4), "rows_out": len(cleaned),
        "reproducible": h1 == h2, "verification_passed": bool((ex.verification or {}).get("passed")),
        "review_items": summary["suggestions_by_status"]["pending"] + summary["anomalies_open"],
        "representation_fixes": len(representation),
        "seconds": round(elapsed, 2),
        "recall_by_kind": {k: f"{a}/{b}" for k, (a, b) in by_kind.items()},
        "missed": [{"row": p.row + 2, "column": p.column, "kind": p.kind, "value": str(p.dirty)}
                   for p in planted if p not in found][:20],
        "false_positives": [{"row": r + 2, "column": c} for r, c in sorted(fp)][:20],
    }


def markdown(results: list[dict], rows: int, seed: int) -> str:
    lines = [f"# IloveCSV benchmark ({rows} clean rows, seed {seed})", "",
             "One synthetic customer dataset with planted problems (whitespace, case variants, placeholders, "
             "out-of-range ages, malformed emails, bad labels, unparseable dates, exact duplicates), written in "
             "every tabular format and prepared with Safe Auto-Clean and no manual decisions.", "",
             "| run | precision | recall | repair accuracy | untouched cells unchanged | valid rows kept | "
             "reproducible | verified | review items | representation fixes | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['format']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['repair_accuracy']:.3f} | "
                     f"{r['untouched_cells_unchanged']:.4f} | {r['valid_rows_preserved']:.3f} | "
                     f"{'yes' if r['reproducible'] else 'NO'} | {'yes' if r['verification_passed'] else 'NO'} | "
                     f"{r['review_items']} | {r['representation_fixes']} | {r['seconds']} |")
    kinds = list(results[0]["recall_by_kind"])
    lines += ["", "Recall by problem type:", "", "| run | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    lines += [f"| {r['format']} | " + " | ".join(r["recall_by_kind"][k] for k in kinds) + " |" for r in results]
    lines += ["", "- **Default runs** use only the rules IloveCSV suggests from the data. With ~2% of ages "
              "impossible, a suggested bound would be broken too often and is dropped on purpose (a bound "
              "violated that often describes the data, it does not validate it), so out-of-range ages need "
              "domain knowledge.",
              "- **csv + domain rule** re-runs the CSV with its saved recipe plus one line a person adds "
              "(`age: {type: int, min: 18, max: 120}`, adult customers): the configurable-rules path.",
              "- **Representation fixes**: valid values held in the wrong representation for the format (ISO "
              "dates typed as text in an Excel sheet, numbers in a text-typed Parquet column). They are real "
              "findings with a safe fix, not planted problems, so they are not counted as false positives."]
    lines += ["", "Out-of-range, malformed and duplicate problems are flagged, not repaired, by design: there "
              "is no safe value to put back. Only whitespace, case variants and placeholders count toward "
              "repair accuracy."]
    return "\n".join(lines) + "\n"


def _print(r: dict) -> None:
    print(f"{r['format']:18} P={r['precision']:.3f} R={r['recall']:.3f} repair={r['repair_accuracy']:.3f} "
          f"untouched={r['untouched_cells_unchanged']:.4f} rows={r['valid_rows_preserved']:.3f} "
          f"repro={r['reproducible']} verified={r['verification_passed']} review={r['review_items']} "
          f"repr={r['representation_fixes']} {r['seconds']}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "benchmark")
    ap.add_argument("--formats", default="csv,tsv,xlsx,json,jsonl,parquet,zip")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    clean = clean_frame(args.rows, rng)
    dirty, planted = plant(clean, rng)
    work = Path(tempfile.mkdtemp(prefix="ilovecsv-bench-"))
    try:
        files = write_formats(dirty, work)
        results = []
        for fmt in args.formats.split(","):
            r = score(fmt, files[fmt], planted, clean.astype(object), dirty, work / f"out_{fmt}")
            results.append(r)
            _print(r)
        if "csv" in files:  # configurable rules: the saved recipe plus one domain rule
            recipe = yaml.safe_load((work / "out_csv" / "customers_prepare_config.yaml").read_text(encoding="utf-8"))
            recipe["columns"]["age"] = {"type": "int", "min": 18, "max": 120}
            cfg = work / "domain.yaml"
            cfg.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
            r = score("csv", files["csv"], planted, clean.astype(object), dirty, work / "out_domain", config=cfg)
            r["format"] = "csv + domain rule"
            results.append(r)
            _print(r)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({"rows": args.rows, "seed": args.seed, "results": results},
                                                      indent=2), encoding="utf-8")
    (args.out / "results.md").write_text(markdown(results, args.rows, args.seed), encoding="utf-8")
    print(f"wrote {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
