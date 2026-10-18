"""FastAPI application.

Run with:  uvicorn ilovecsv.api.main:app --port 8010
"""

from __future__ import annotations

from fastapi import FastAPI

from .. import __version__
from ..db import make_engine, make_sessionmaker
from . import errors
from .routes import runs, workbooks
from .settings import Settings
from .uploads import BodySizeLimit


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(
        title="IloveCSV API",
        version=__version__,
        description="Spreadsheet quality and validation engine: upload, profile, validate, review "
                    "suggested corrections, export a corrected workbook and an audit report.",
    )
    engine = make_engine(settings.db_url)
    app.state.settings = settings
    app.state.engine = engine
    app.state.sessionmaker = make_sessionmaker(engine)
    errors.install(app)
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_upload_bytes)
    app.include_router(workbooks.router)
    app.include_router(runs.router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "service": "ilovecsv", "version": __version__}

    return app


app = create_app()
