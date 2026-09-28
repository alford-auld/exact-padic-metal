"""Locate and import the Yates butterfly kernel.

The kernel is vendored at ``vendor/yates`` -- a verbatim copy of the ``yates``
package from *exact-yates-metal*, pinned by ``vendor/KERNEL_COMMIT``.  It is
copied in rather than referenced as a submodule because every run executes an
immutable source snapshot of the recorded commit, and those snapshots do not
carry submodule contents.  See ``vendor/PROVENANCE.md``.

``PADIC_YATES_PATH`` points at a directory *containing* a ``yates`` package,
for developing against an upstream checkout instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _ROOT / "vendor"


def yates_root() -> Path:
    """Directory to put on ``sys.path`` so that ``import yates`` works."""
    override = os.environ.get("PADIC_YATES_PATH")
    candidates = [Path(override)] if override else []
    candidates.append(_VENDOR)
    for root in candidates:
        if (root / "yates" / "__init__.py").is_file():
            return root
    raise ImportError(
        f"no `yates` package found in {[str(c) for c in candidates]}.\n"
        "The kernel should be vendored at vendor/yates; see vendor/PROVENANCE.md, "
        "or set PADIC_YATES_PATH to a directory containing a `yates` package."
    )


def load():
    """Import and return the ``yates`` package."""
    root = str(yates_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import yates  # noqa: PLC0415

    return yates


def kernel_commit() -> str:
    """The upstream commit the vendored copy was taken from."""
    try:
        return (_VENDOR / "KERNEL_COMMIT").read_text().strip()[:12]
    except OSError:
        return "unknown"
