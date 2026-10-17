import os
import tempfile
from pathlib import Path

import pytest

# Keep the API's import-time app from writing into the repo.
os.environ.setdefault("ILOVECSV_DATA_DIR", str(Path(tempfile.gettempdir()) / "ilovecsv-test-default"))

from ilovecsv.synth import RULES_YAML, build_edge_workbook, build_orders_workbook  # noqa: E402


@pytest.fixture(scope="session")
def rules_text() -> str:
    return RULES_YAML


@pytest.fixture(scope="session")
def orders(tmp_path_factory):
    """Injected workbook and its ground-truth manifest."""
    path = tmp_path_factory.mktemp("wb") / "orders.xlsx"
    manifest = build_orders_workbook(path, n=300, seed=7, inject=True)
    return path, manifest


@pytest.fixture(scope="session")
def clean_orders(tmp_path_factory):
    path = tmp_path_factory.mktemp("wb") / "orders_clean.xlsx"
    build_orders_workbook(path, n=300, seed=7, inject=False)
    return path


@pytest.fixture(scope="session")
def edge_xlsx(tmp_path_factory):
    path = tmp_path_factory.mktemp("wb") / "edge.xlsx"
    build_edge_workbook(path)
    return path


@pytest.fixture(scope="session")
def orders_run(orders, rules_text):
    from ilovecsv.pipeline import run_checks

    return run_checks(orders[0], rules_text)
