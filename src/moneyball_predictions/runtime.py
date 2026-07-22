"""Runtime helpers for local development and serverless deployment."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path


def running_on_vercel() -> bool:
    """Return whether the process is running in a Vercel-like function runtime.

    Vercel's automatically exposed system variables can be disabled at the
    project level, so relying on only ``VERCEL`` is brittle. We also honor an
    explicit application flag and detect the standard ``/var/task`` function
    root used by the Python runtime.
    """
    explicit = os.environ.get("MONEYBALL_SERVERLESS", "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if any(
        os.environ.get(name)
        for name in ("VERCEL", "VERCEL_ENV", "VERCEL_URL", "VERCEL_REGION")
    ):
        return True
    try:
        return Path.cwd().resolve().as_posix().startswith("/var/task") or Path("/var/task").exists()
    except OSError:
        return False


def _ensure_writable(directory: Path) -> Path:
    """Create a directory and verify it is writable."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".moneyball-write-{uuid.uuid4().hex}"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)
    return directory


def runtime_cache_dir(namespace: str) -> Path:
    """Return a guaranteed writable cache directory.

    Vercel deployments are read-only except for ``/tmp``. This function does
    not trust runtime detection alone: it probes the preferred location and
    falls back to the operating system temporary directory on any write error.
    """
    override = os.environ.get("MONEYBALL_RUNTIME_CACHE_DIR")
    if override:
        preferred_root = Path(override)
    elif running_on_vercel():
        preferred_root = Path("/tmp/moneyball")
    else:
        preferred_root = Path("data")

    preferred = preferred_root / namespace
    try:
        return _ensure_writable(preferred)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "moneyball" / namespace
        return _ensure_writable(fallback)
