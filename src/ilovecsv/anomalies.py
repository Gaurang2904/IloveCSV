"""Anomaly detection: values that look suspicious but break no stated rule.

Anomalies are deliberately kept apart from errors:

* a cell that already has a rule error is not flagged again as an anomaly;
* anomalies start with status "open" and a person decides whether each one is
  a confirmed error or should be dismissed;
* anomalies never get automatic correction suggestions.

Detectors:
  robust_z          |0.6745 * (x - median) / MAD| > z_threshold, per numeric column
  iqr               fallback when MAD is 0: outside Q1 - 3*IQR .. Q3 + 3*IQR
  rare_category     a value seen in < rare_threshold of rows (count <= 2) in a
                    low-cardinality text column with no `allowed` rule
  isolation_forest  row-level, on the numeric columns together (>= if_min_rows rows),
                    after PCA whitening; flagged when the isolation score's robust
                    z is below -if_z_threshold
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .models import ANOMALY, ERROR, Issue
from .profiling import SheetProfile
from .reader import WorkbookData
from .rules.schema import AnomalyConfig, RuleSet, SheetRules
from .values import normalise_nulls, to_number, to_text

NUMERIC = ("int", "float")
MAX_FLAG_SHARE = 0.01  # more than 1% of a column flagged -> report the distribution, not each value
MAX_FLAGS_ABS = 5


def detect_anomalies(
    wb: WorkbookData,
    rules: RuleSet,
    profiles: dict[str, SheetProfile],
    existing: list[Issue],
) -> list[Issue]:
    cfg = rules.anomalies
    if not cfg.enabled:
        return []
    nulls = normalise_nulls(rules.null_values)
    error_cells = {i.cell_ref for i in existing if i.kind == ERROR and i.cell_ref}
    error_rows = {(i.sheet, i.xl_row) for i in existing if i.kind == ERROR and i.column is None}
    out: list[Issue] = []
    for srules in rules.sheets:
        name = srules.sheet or wb.first_sheet.name
        sheet = wb.sheets.get(name)
        if sheet is None:
            continue
        prof = profiles[name]
        cols = _numeric_columns(prof, srules, cfg)
        numeric = {
            c: pd.Series([to_number(v, nulls) for v in sheet.df[c]], index=sheet.df.index, dtype="float64")
            for c in cols
        }
        flagged: set[str] = set()

        for col, s in numeric.items():
            hits = [h for h in _univariate(s, cfg) if sheet.cell_ref(h[0], col) not in error_cells]
            n = int(s.notna().sum())
            if len(hits) > max(MAX_FLAGS_ABS, MAX_FLAG_SHARE * n):
                # Anomalies are rare by definition. When many values cross the threshold the column is
                # heavy-tailed or multi-modal (common in measurement signals), and per-value flags would
                # be noise: report the distribution once instead.
                out.append(Issue(ANOMALY, name, "anomaly.distribution", "info",
                                 f"'{col}': {len(hits):,} of {n:,} values ({len(hits) / n:.1%}) are far from the "
                                 "median by the robust z rule; the distribution is heavy-tailed or multi-modal, "
                                 "so per-value outlier flags are suppressed. Check its units and expected "
                                 "range in the documentation.", column=col, detector="distribution",
                                 score=round(len(hits) / n, 4)))
                continue
            for xl_row, score, detector, msg in hits:
                ref = sheet.cell_ref(xl_row, col)
                flagged.add(ref)
                out.append(Issue(ANOMALY, name, f"anomaly.{detector}", "info", msg, xl_row=xl_row,
                                 column=col, cell_ref=ref, value=sheet.df.at[xl_row, col],
                                 detector=detector, score=round(score, 3)))

        for col in _category_columns(prof, srules, cfg):
            values = [to_text(v, nulls) for v in sheet.df[col]]
            present = [v for v in values if v is not None]
            counts = Counter(present)
            n = len(present)
            for xl_row, v in zip(sheet.df.index, values, strict=True):
                if v is None:
                    continue
                share = counts[v] / n
                if share < cfg.rare_threshold and counts[v] <= 2:
                    ref = sheet.cell_ref(int(xl_row), col)
                    if ref in error_cells:
                        continue
                    flagged.add(ref)
                    out.append(Issue(ANOMALY, name, "anomaly.rare_category", "info",
                                     f"'{v}' appears only {counts[v]} time(s) in {n} rows "
                                     f"({share:.2%}); possible typo or one-off category.",
                                     xl_row=int(xl_row), column=col, cell_ref=ref, value=v,
                                     detector="rare_category", score=round(share, 4)))

        if cfg.isolation_forest and len(numeric) >= 2 and len(sheet.df) >= cfg.if_min_rows:
            rows = []
            for xl_row, score, top_col, z in _isolation_forest(numeric, cfg):
                ref = sheet.cell_ref(xl_row, top_col)
                if ref in error_cells or ref in flagged or (name, xl_row) in error_rows:
                    continue
                if any(sheet.cell_ref(xl_row, c) in error_cells for c in numeric):
                    continue  # the row already has a confirmed numeric error
                rows.append((xl_row, score, top_col, z, ref))
            if len(rows) > max(MAX_FLAGS_ABS, MAX_FLAG_SHARE * len(sheet.df)):
                out.append(Issue(ANOMALY, name, "anomaly.distribution", "info",
                                 f"{len(rows):,} rows ({len(rows) / len(sheet.df):.1%}) stand out to the Isolation "
                                 "Forest: the numeric columns form several regimes (e.g. different experiments, "
                                 "cycle types or units), so per-row flags are suppressed. Consider preparing each "
                                 "regime separately.", detector="isolation_forest_distribution",
                                 score=round(len(rows) / len(sheet.df), 4)))
                rows = []
            for xl_row, score, top_col, z, ref in rows:
                ctx = ", ".join(f"{c}={_fmt(numeric[c].loc[xl_row])}" for c in numeric)
                out.append(Issue(ANOMALY, name, "anomaly.isolation_forest", "info",
                                 f"Unusual combination of values ({ctx}); isolation score "
                                 f"z = {score:+.1f}. Largest single deviation: {top_col} (z {z:+.1f}).",
                                 xl_row=xl_row, column=top_col, cell_ref=ref,
                                 value=sheet.df.at[xl_row, top_col],
                                 detector="isolation_forest", score=round(abs(score), 3)))
    return out


def _numeric_columns(prof: SheetProfile, srules: SheetRules, cfg: AnomalyConfig) -> list[str]:
    cols = []
    for name, c in prof.columns.items():
        declared = srules.columns.get(name)
        ctype = declared.type if declared and declared.type else c.inferred_type
        if ctype not in ("int", "float", "number") or c.is_identifier or name in cfg.exclude:
            continue
        if name == srules.target:
            continue  # a rare label (fraud, churn) is the signal, not an anomaly
        if cfg.columns is not None and name not in cfg.columns:
            continue
        if c.n_rows - c.n_missing < 10:
            continue
        cols.append(name)
    return cols


def _category_columns(prof: SheetProfile, srules: SheetRules, cfg: AnomalyConfig) -> list[str]:
    cols = []
    for name, c in prof.columns.items():
        declared = srules.columns.get(name)
        if declared and declared.allowed is not None:
            continue  # the allowed rule already decides validity
        if name == srules.target:
            continue  # a rare class is the signal, not an anomaly
        if name in cfg.exclude or c.is_identifier:
            continue
        non_null = c.n_rows - c.n_missing
        if c.inferred_type in ("category", "text") and 2 <= c.n_distinct <= 20 and non_null >= 50:
            cols.append(name)
    return cols


def _robust_z(s: pd.Series) -> tuple[pd.Series, float]:
    x = s.dropna()
    med = float(x.median())
    mad = float((x - med).abs().median())
    if mad == 0:
        return pd.Series(np.nan, index=s.index), 0.0
    return 0.6745 * (s - med) / mad, mad


def _univariate(s: pd.Series, cfg: AnomalyConfig):
    x = s.dropna()
    if x.size < 10:
        return
    z, mad = _robust_z(s)
    med = float(x.median())
    if mad > 0:
        for xl_row, zv in z.dropna().items():
            if abs(zv) > cfg.z_threshold:
                yield (int(xl_row), abs(float(zv)), "robust_z",
                       f"{_fmt(s.loc[xl_row])} is far from the typical value {_fmt(med)} "
                       f"(robust z = {zv:+.1f}, threshold {cfg.z_threshold}).")
        return
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        # Almost all values identical: anything different is unusual.
        common = float(x.mode().iloc[0])
        share = float((x == common).mean())
        if share >= 0.95:
            for xl_row, v in x.items():
                if v != common:
                    yield (int(xl_row), share, "iqr",
                           f"{_fmt(v)} differs from the value {_fmt(common)} used in {share:.0%} of rows.")
        return
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    for xl_row, v in x.items():
        if v < lo or v > hi:
            yield (int(xl_row), float(abs(v - med) / iqr), "iqr",
                   f"{_fmt(v)} is outside the far fences {_fmt(lo)}..{_fmt(hi)} (3 x IQR).")


def _isolation_forest(numeric: dict[str, pd.Series], cfg: AnomalyConfig):
    """Row-level detector for unusual *combinations* of values.

    Columns are robust-scaled, then whitened with PCA fitted on the central 90%
    of rows. Whitening matters: Isolation Forest splits along axes, so a row that
    breaks a correlation (high quantity, tiny shipping cost) only becomes easy to
    isolate once the correlated directions are rotated onto their own axis.
    Isolation scores are then compared with each other (robust z of the score),
    so the threshold does not depend on the data's scale or row count.
    """
    frame = pd.DataFrame(numeric)
    frame = frame.loc[frame.notna().sum(axis=1) >= max(2, int(0.5 * frame.shape[1]))]
    if len(frame) < cfg.if_min_rows:
        return
    filled = frame.fillna(frame.median())
    med = filled.median()
    scale = (filled - med).abs().median().replace(0, np.nan)
    scale = scale.fillna(filled.std(ddof=0).replace(0, 1.0))
    X = ((filled - med) / scale).to_numpy()
    spread = np.abs(X).max(axis=1)
    core = X[spread <= np.quantile(spread, 0.9)]
    W = PCA(whiten=True, random_state=0).fit(core).transform(X)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(W)
    scores = model.score_samples(W)  # lower = easier to isolate
    s_med = float(np.median(scores))
    s_mad = float(np.median(np.abs(scores - s_med))) or 1e-9
    score_z = 0.6745 * (scores - s_med) / s_mad
    zs = 0.6745 * (filled - med) / scale
    for i in np.where(score_z < -cfg.if_z_threshold)[0]:
        xl_row = int(frame.index[i])
        row_z = zs.iloc[i]
        top_col = str(row_z.abs().idxmax())
        yield xl_row, float(score_z[i]), top_col, float(row_z[top_col])


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "null"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.2f}".rstrip("0").rstrip(".") if not float(v).is_integer() else f"{int(v):,}"
    return str(v)
