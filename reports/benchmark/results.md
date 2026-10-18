# IloveCSV benchmark (2000 clean rows, seed 7)

One synthetic customer dataset with planted problems (whitespace, case variants, placeholders, out-of-range ages, malformed emails, bad labels, unparseable dates, exact duplicates), written in every tabular format and prepared with Safe Auto-Clean and no manual decisions.

| run | precision | recall | repair accuracy | untouched cells unchanged | valid rows kept | reproducible | verified | review items | representation fixes | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| csv | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 22 | 0 | 1.15 |
| tsv | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 22 | 0 | 1.13 |
| xlsx | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 22 | 2000 | 3.32 |
| json | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 22 | 0 | 1.1 |
| jsonl | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 22 | 0 | 1.13 |
| parquet | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 21 | 3980 | 2.57 |
| zip | 1.000 | 0.879 | 1.000 | 1.0000 | 1.000 | yes | yes | 22 | 0 | 1.22 |
| csv + domain rule | 1.000 | 1.000 | 1.000 | 1.0000 | 1.000 | yes | yes | 20 | 0 | 1.11 |

Recall by problem type:

| run | whitespace | case_variant | placeholder | out_of_range | invalid_email | bad_label | invalid_date | exact_duplicate |
|---|---|---|---|---|---|---|---|---|
| csv | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| tsv | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| xlsx | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| json | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| jsonl | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| parquet | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| zip | 40/40 | 40/40 | 40/40 | 6/40 | 40/40 | 40/40 | 20/20 | 20/20 |
| csv + domain rule | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 20/20 | 20/20 |

- **Default runs** use only the rules IloveCSV suggests from the data. With ~2% of ages impossible, a suggested bound would be broken too often and is dropped on purpose (a bound violated that often describes the data, it does not validate it), so out-of-range ages need domain knowledge.
- **csv + domain rule** re-runs the CSV with its saved recipe plus one line a person adds (`age: {type: int, min: 18, max: 120}`, adult customers): the configurable-rules path.
- **Representation fixes**: valid values held in the wrong representation for the format (ISO dates typed as text in an Excel sheet, numbers in a text-typed Parquet column). They are real findings with a safe fix, not planted problems, so they are not counted as false positives.

Out-of-range, malformed and duplicate problems are flagged, not repaired, by design: there is no safe value to put back. Only whitespace, case variants and placeholders count toward repair accuracy.
