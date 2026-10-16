"""Runtime dataset download with pinned revisions and SHA-256 verification.

Files are cached outside the repository (``$WORKFORCE_DATA_DIR`` or
``~/.cache/workforce/datasets``) and are never committed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class RemoteFile:
    url: str
    sha256: str
    filename: str


def data_dir() -> Path:
    root = os.environ.get("WORKFORCE_DATA_DIR")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "workforce" / "datasets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(remote: RemoteFile, attempts: int = 4) -> Path:
    target = data_dir() / remote.filename
    if target.is_file() and _sha256(target) == remote.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            tmp_path = _download(remote.url, target.parent)
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            transient = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code >= 500 or exc.response.status_code == 429
            )
            if not transient or attempt == attempts:
                raise
            time.sleep(2**attempt)
    actual = _sha256(tmp_path)
    if actual != remote.sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {remote.url}: expected {remote.sha256}, got {actual}"
        )
    tmp_path.replace(target)
    return target


def _download(url: str, directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    tmp.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return tmp_path
