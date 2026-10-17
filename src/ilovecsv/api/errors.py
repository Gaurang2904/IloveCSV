"""RFC 9457 problem details (application/problem+json) for every error response."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("ilovecsv.api")
PROBLEM_JSON = "application/problem+json"


class Problem(StarletteHTTPException):
    """Raise anywhere in the API to return a structured error."""

    def __init__(self, status: int, slug: str, detail: str, errors: list[dict[str, Any]] | None = None,
                 title: str | None = None, headers: dict[str, str] | None = None):
        super().__init__(status_code=status, detail=detail, headers=headers)
        self.slug = slug
        self.title = title or HTTPStatus(status).phrase
        self.errors = errors or []


def problem_body(status: int, slug: str, title: str, detail: str, instance: str,
                 errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"type": f"/problems/{slug}", "title": title, "status": status,
                            "detail": detail, "instance": instance}
    if errors:
        body["errors"] = errors
    return body


def problem_response(request: Request, status: int, slug: str, detail: str, title: str | None = None,
                     errors: list[dict[str, Any]] | None = None, headers: dict[str, str] | None = None) -> JSONResponse:
    body = problem_body(status, slug, title or HTTPStatus(status).phrase, detail, request.url.path, errors)
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


_SLUGS = {400: "bad-request", 401: "unauthorized", 403: "forbidden", 404: "not-found",
          405: "method-not-allowed", 409: "conflict", 413: "file-too-large",
          415: "unsupported-media-type", 422: "validation-error", 429: "too-many-requests"}


def install(app: FastAPI) -> None:
    @app.exception_handler(Problem)
    async def _problem(request: Request, exc: Problem):
        return problem_response(request, exc.status_code, exc.slug, str(exc.detail), exc.title,
                                exc.errors, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        slug = _SLUGS.get(exc.status_code, "http-error")
        detail = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
        return problem_response(request, exc.status_code, slug, detail, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errors = [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg", "")} for e in exc.errors()]
        return problem_response(request, 422, "request-validation", "The request is not valid.",
                                "Unprocessable Content", errors)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error on %s", request.url.path)
        return problem_response(request, 500, "internal-error",
                                "An unexpected error occurred. The details were logged on the server.")
