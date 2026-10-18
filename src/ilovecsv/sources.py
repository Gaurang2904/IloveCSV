"""How a source file is turned into tables: options that belong in the recipe.

Every option here changes *what the tables are* (which sheet, which JSON array, which members
of a ZIP), so it is saved with the rules and applied again when the recipe is reused.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_ROWS = 1_000_000  # full in-memory processing limit per table; larger needs --max-rows


class SourceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheets: list[str] | None = Field(default=None, description="Excel sheets to process (default: all with data).")
    tables: list[str] | None = Field(
        default=None, description="Tables to process from an archive or a scientific file (default: all).")
    delimiter: str | None = Field(default=None, max_length=1, description="CSV delimiter (default: detected).")
    encoding: str | None = Field(default=None, description="Text encoding (default: utf-8, then cp1252).")
    json_records: str | None = Field(
        default=None, description="Dot path to the array of records in a JSON file, e.g. 'data.items' "
                                  "(default: the top-level array, or the only array of objects).")
    max_rows: int | None = Field(
        default=None, ge=1, description="Process only the first N rows of each table (a sample; marked as partial).")
    combine_tables: bool = Field(
        default=True, description="In archives, merge same-named tables with identical columns (adds source_file).")
    align_arrays: bool = Field(
        default=False, description="Combine same-length 1-D arrays (NumPy, HDF5 group, MATLAB variables) into one "
                                   "table. Off by default: equal length alone does not prove the arrays are aligned.")
    series_context: list[str] | None = Field(
        default=None, description="MATLAB struct arrays: numeric per-element fields (e.g. ambient_temperature) to "
                                  "repeat on every row of the series table. Text fields are always repeated; "
                                  "numbers are not by default, since one may be the outcome (e.g. capacity).")
    mat_layout: Literal["auto", "nasa_battery", "generic"] = Field(
        default="auto", description="How MATLAB files become tables: auto-detect a known layout, or generic.")
    row_limit: int = Field(default=DEFAULT_MAX_ROWS, ge=1,
                           description="Refuse full processing above this many rows unless max_rows is set.")
