"""Locate and import the Yates butterfly kernel.

The kernel lives in ``vendor/exact-yates-metal``, a git submodule pinned to the
commit the measurements in this repository were taken with.  It ships no
``pyproject.toml``, so it is imported by path rather than installed.
``PADIC_YATES_PATH`` overrides the location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent / "vendor" / "exact-yates-metal"


def yates_root() -> Path:
    root = Path(os.environ.get("PADIC_YATES_PATH", _DEFAULT))
    if not (root / "yates" / "__init__.py").is_file():
        raise ImportError(
            f"the Yates kernel is not at {root}.\n"
            "Run `git submodule update --init` in the repository root, or set "
            "PADIC_YATES_PATH to a checkout of exact-yates-metal."
        )
    return root


def load():
    """Import and return the ``yates`` package."""
    root = str(yates_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import yates  # noqa: PLC0415

    return yates


def kernel_commit() -> str:
    """The submodule commit, for provenance in run logs."""
    head = yates_root() / ".git"
    try:
        if head.is_file():  # submodule: a gitdir pointer
            gitdir = Path(head.read_text().split("gitdir:", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (yates_root() / gitdir).resolve()
            head = gitdir
        ref = (head / "HEAD").read_text().strip()
        if ref.startswith("ref:"):
            ref = (head / ref.split(" ", 1)[1]).read_text().strip()
        return ref[:12]
    except Exception:
        return "unknown"
