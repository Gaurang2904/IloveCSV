"""Upload safety: body-size limit, type sniffing, zip-bomb and macro checks."""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

from fastapi import UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import PROBLEM_JSON, Problem, problem_body
from .settings import Settings

ALLOWED_EXT = {".xlsx", ".csv"}
CHUNK = 1 << 16


class BodySizeLimit:
    """Reject request bodies over the limit, from Content-Length or while streaming.

    Checking Content-Length stops honest clients before any bytes are read;
    counting received bytes stops clients that lie or use chunked encoding.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, overhead: int = 64 * 1024):
        self.app, self.max_bytes = app, max_bytes
        self.limit = max_bytes + overhead  # overhead: multipart boundaries + form fields

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            await _send_413(scope, send, self.max_bytes)
            return
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise Problem(413, "file-too-large", _too_large(self.max_bytes))
            return message

        await self.app(scope, limited, send)


def _too_large(limit: int) -> str:
    return f"The upload exceeds the {limit / (1024 * 1024):g} MB limit."


async def _send_413(scope: Scope, send: Send, limit: int) -> None:
    body = json.dumps(problem_body(413, "file-too-large", "Content Too Large", _too_large(limit),
                                   scope.get("path", ""))).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", PROBLEM_JSON.encode()), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def save_upload(file: UploadFile, settings: Settings) -> tuple[Path, int, str]:
    """Validate and store an upload under a random name. Returns (path, size, extension)."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext in (".xls", ".xlsm", ".xlsb"):
        raise Problem(415, "unsupported-media-type",
                      f"'{ext}' files are not accepted. Save the workbook as .xlsx (macro-free) or .csv.")
    if ext not in ALLOWED_EXT:
        raise Problem(415, "unsupported-media-type", "Only .xlsx and .csv files are accepted.")
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise Problem(413, "file-too-large",
                                  f"The file exceeds the {settings.max_upload_mb:g} MB limit.")
                out.write(chunk)
        if size == 0:
            raise Problem(422, "empty-file", "The uploaded file is empty.")
        _sniff(dest, ext, settings)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest, size, ext


def _sniff(path: Path, ext: str, settings: Settings) -> None:
    with path.open("rb") as f:
        head = f.read(8192)
    if ext == ".csv":
        if b"\x00" in head or head.startswith(b"PK\x03\x04"):
            raise Problem(415, "content-mismatch", "The file content is not plain-text CSV.")
        return
    if not head.startswith(b"PK\x03\x04"):
        raise Problem(415, "content-mismatch", "The file content is not an .xlsx workbook.")
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            names = {i.filename for i in infos}
    except zipfile.BadZipFile as e:
        raise Problem(415, "content-mismatch", "The .xlsx file is corrupt.") from e
    if len(infos) > settings.max_zip_entries:
        raise Problem(422, "unsafe-archive", "The workbook contains too many internal parts.")
    total = sum(i.file_size for i in infos)
    if total > settings.max_uncompressed_mb * 1024 * 1024:
        raise Problem(422, "unsafe-archive", "The workbook expands to more than the allowed size.")
    for i in infos:
        if i.compress_size and i.file_size / i.compress_size > settings.max_compression_ratio and i.file_size > 1 << 20:
            raise Problem(422, "unsafe-archive", "The workbook has a suspicious compression ratio.")
    if "xl/vbaProject.bin" in names:
        raise Problem(415, "unsupported-media-type", "Macro-enabled workbooks are not accepted.")
    if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
        raise Problem(415, "content-mismatch", "The file is a zip archive but not an Excel workbook.")
