"""Runtime helpers for local development and serverless deployment."""

from __future__ import annotations

import os
from pathlib import Path


def running_on_vercel() -> bool:
    """Return whether the process is running inside Vercel."""
    return bool(os.environ.get("VERCEL"))


def runtime_cache_dir(namespace: str) -> Path:
    """Return a writable cache directory for the current runtime.

    Vercel's deployed bundle is read-only, while ``/tmp`` is writable for the
    lifetime of a function instance. Local development keeps using ``data``.
    """
    override = os.environ.get("MONEYBALL_RUNTIME_CACHE_DIR")
    if override:
        root = Path(override)
    elif running_on_vercel():
        root = Path("/tmp/moneyball")
    else:
        root = Path("data")
    return root / namespace
