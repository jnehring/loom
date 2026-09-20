"""Resolve Loom's on-disk state directories.

Resolution order for the home directory:

1. Explicit ``home`` argument (when a helper accepts one)
2. ``$LOOM_HOME`` environment variable
3. ``~/.loom``

The response cache additionally honours ``$LOOM_CACHE_DIR`` before falling
back to ``<loom_home>/cache``.
"""

from __future__ import annotations

import os
from pathlib import Path


def loom_home() -> Path:
    """Return the Loom state directory (``$LOOM_HOME`` or ``~/.loom``)."""
    env = os.environ.get("LOOM_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".loom"


def default_cache_dir() -> Path:
    """Return the response-cache directory.

    Order: ``$LOOM_CACHE_DIR`` → ``<loom_home>/cache``.
    """
    env = os.environ.get("LOOM_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return loom_home() / "cache"


def default_batches_dir() -> Path:
    """Return the batch-metadata directory: ``<loom_home>/batches``."""
    return loom_home() / "batches"


def default_inputs_dir() -> Path:
    """Return the directory used for in-memory batch prompt snapshots."""
    return loom_home() / "inputs"
