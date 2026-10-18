"""SQLite persistence (SQLAlchemy 2.0): users, workbooks, runs, issues, suggestions, audit trail."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utcnow)


class Workbook(Base):
    __tablename__ = "workbooks"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(8))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sheets: Mapped[Any] = mapped_column(JSON)
    warnings: Mapped[Any] = mapped_column(JSON, default=list)
    uploaded_at: Mapped[str] = mapped_column(String(32), default=utcnow)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workbook_id: Mapped[str] = mapped_column(ForeignKey("workbooks.id"), index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    rules_text: Mapped[str] = mapped_column(Text)
    rules_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(32), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="validated")  # validated | exported
    summary: Mapped[Any] = mapped_column(JSON)
    exported_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    out_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    corrected_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    reports: Mapped[Any] = mapped_column(JSON, nullable=True)
    after: Mapped[Any] = mapped_column(JSON, nullable=True)


class IssueRow(Base):
    __tablename__ = "issues"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(10), index=True)
    sheet: Mapped[str] = mapped_column(String(100))
    xl_row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cell_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    rule_id: Mapped[str] = mapped_column(String(100), index=True)
    severity: Mapped[str] = mapped_column(String(10))
    value: Mapped[Any] = mapped_column(JSON, nullable=True)
    message: Mapped[str] = mapped_column(Text)
    detector: Mapped[str | None] = mapped_column(String(50), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="open")
    decided_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    decided_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[Any] = mapped_column(JSON, nullable=True)
    introduced_by_fix: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    corrected_cell_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)


class SuggestionRow(Base):
    __tablename__ = "suggestions"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    issue_id: Mapped[str] = mapped_column(String(32), index=True)
    sheet: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(16))
    method: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)
    xl_row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cell_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    old: Mapped[Any] = mapped_column(JSON, nullable=True)
    new: Mapped[Any] = mapped_column(JSON, nullable=True)
    rule_id: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(10), default="pending", index=True)
    decided_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    decided_at: Mapped[str | None] = mapped_column(String(32), nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[str] = mapped_column(String(32), default=utcnow)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    workbook_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(50))
    detail: Mapped[Any] = mapped_column(JSON, nullable=True)


def make_engine(url: str) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover - trivial
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> None:
    """Tiny forward-only migration: add columns introduced after a database was created."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing:
                    ddl = col.type.compile(engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))


def make_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
