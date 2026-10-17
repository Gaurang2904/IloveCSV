"""Shared dependencies: DB sessions, API-key authentication, audit logging."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import AuditEvent, User
from .errors import Problem

KEY_PREFIX = "ilc_"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key from scripts/create_api_key.py")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def create_user(session: Session, name: str) -> str:
    """Create a user and return their plain API key (only the hash is stored)."""
    key = new_api_key()
    session.add(User(name=name, key_hash=hash_key(key)))
    session.commit()
    return key


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.sessionmaker() as session:
        yield session


def current_user(
    key: str | None = Depends(api_key_header), db: Session = Depends(get_db)
) -> User:
    challenge = {"WWW-Authenticate": "ApiKey"}
    if not key:
        raise Problem(401, "missing-api-key", "Send your API key in the X-API-Key header.", headers=challenge)
    digest = hash_key(key)
    user = db.scalar(select(User).where(User.key_hash == digest))
    if user is None or not hmac.compare_digest(user.key_hash, digest):
        raise Problem(401, "invalid-api-key", "The API key is not valid.", headers=challenge)
    return user


def audit(db: Session, user: User | None, action: str, *, workbook_id: str | None = None,
          run_id: str | None = None, detail: dict[str, Any] | None = None) -> None:
    db.add(AuditEvent(user_id=user.id if user else None, action=action, workbook_id=workbook_id,
                      run_id=run_id, detail=detail))
