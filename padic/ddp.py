"""Algorithm 6: digit dynamic programming for ``e_max``, at p = 2.

Algorithm 6 of arXiv:2603.29905v1 maintains ``Z``, the common zeros of the
system modulo ``p^e``, and lifts each ``z`` by every digit vector
``d in N_{<p}^L`` to ``w = z + p^e d``, keeping those that vanish modulo
``p^(e+1)``.  At ``p = 2`` the digit vector is a **subset** of the ``L``
coordinates, so one level of the dynamic program is a computation over the
Boolean lattice ``2^[L]`` -- the native domain of the Yates butterfly kernel.

Two paths compute a level, and they are checked against each other:

``dense``
    ``f_i(z + 2^e d)`` collapses to a multilinear polynomial in ``d``
    (``poly.lift_coefficients``), and the subset-zeta transform of its
    coefficient vector is the table of all ``2^L`` values.  One batched
    ``ZETA_SUB`` per chunk of ``z`` replaces the whole of lines 7-12.  This is
    Algorithm 6 as written, at memory bandwidth.

``structural``
    For ``e >= 1`` only.  Since ``d_l in {0,1}``, terms of the lift with two
    or more digit factors carry ``2^(2e)``, and ``2e >= e+1``, so the lift is
    **affine** in ``d``.  Writing ``f_i(z) = 2^e c_i`` (legitimate: ``z`` is a
    zero modulo ``2^e``), the surviving ``d`` are exactly the solutions of

        sum_l g_{i,l} d_l = c_i  over F_2,   g_{i,l} = (coeff of d_l) / 2^e,

    an affine system solvable in ``O(|I| L^2)`` instead of ``2^L``.

At ``e = 0`` there is no such shortcut and none is expected: the step asks for
the common roots in ``{0,1}^L`` of a system of multilinear polynomials over
F_2, which is NP-hard.  The transform is load-bearing exactly where the
problem is hard.

Guarantees
----------
Two effects can stop the dynamic program before it has proved anything about
larger ``e``, and both can only make the answer **too small**:

* the level bound ``max_e`` (the paper's own remedy for the ``while True``
  that never halts when a zero exists in ``Z_p``), and
* the survivor cap, which discards elements of ``Z`` that might have lifted.

So a capped run reports ``guarantee = "lower_bound"``: there *is* a common
zero modulo ``2^e_max``, and nothing is claimed above it.  An uncapped run
that ends because ``W`` was genuinely empty reports ``guarantee = "exact"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Literal, Sequence

import numpy as np

from . import _yates
from .gf2 import solve_affine
from .poly import Poly, RING_BITS, eval_mod, lift_coefficients

Backend = Literal["dense", "structural", "numpy"]

#: Bytes of coefficient table to build per chunk.  8 GiB of unified memory on
#: the smallest supported machine, of which this is a conservative slice.
DEFAULT_BUDGET_BYTES = 1 << 30

#: Above this, the F_2 row bitmasks stop fitting the vectorised path.
MAX_STRUCTURAL_VARS = 62


@dataclass
class LevelStat:
    """What one iteration of the while loop did."""

    e: int
    backend: str
    survivors_in: int
    survivors_out: int
    truncated: bool
    seconds: float


@dataclass
class Result:
    e_max: int
    guarantee: Literal["exact", "lower_bound"]
    reason: str
    levels: List[LevelStat] = field(default_factory=list)
    kernel_commit: str = ""
    #: One common zero modulo ``2^e_max``, if the level was ever reached.
    #: ``e_max = 0`` makes this the zero vector, which is a zero modulo
    #: ``2^0 = 1`` vacuously -- check ``e_max`` before using it.
    witness: List[int] | None = None

    @property
    def exact(self) -> bool:
        return self.guarantee == "exact"

    def summary(self) -> str:
        rel = "=" if self.exact else ">="
        return f"e_max {rel} {self.e_max}  ({self.reason})"


# -- helpers ---------------------------------------------------------------


def _digits(masks: np.ndarray, L: int) -> np.ndarray:
    """``(K,)`` subset bitmasks -> ``(K, L)`` uint64 digit vectors."""
    shifts = np.arange(L, dtype=np.uint64)
    return ((masks.astype(np.uint64)[:, None] >> shifts) & np.uint64(1)).astype(np.uint64)


def zeta_sub_numpy(c: np.ndarray) -> np.ndarray:
    """Subset zeta along the last axis, in uint64 -- the CPU reference path."""
    n = c.shape[-1].bit_length() - 1
    out = c.copy()
    for j in range(n):
        step = 1 << j
        idx = (np.arange(out.shape[-1]) >> j) & 1 == 1
        out[..., idx] += out[..., np.nonzero(idx)[0] - step]
    return out


# -- one level, dense ------------------------------------------------------


def _level_dense(
    polys: Sequence[Poly],
    Z: np.ndarray,
    e: int,
    L: int,
    *,
    use_gpu: bool,
    budget_bytes: int,
    cap: int,
) -> tuple[np.ndarray, bool]:
    """Lift every ``z`` by every ``d`` via one subset-zeta per chunk."""
    N = 1 << L
    modmask = np.uint64((1 << (e + 1)) - 1) if e + 1 < 64 else np.uint64((1 << 64) - 1)

    if use_gpu:
        import mlx.core as mx  # noqa: PLC0415

        yates = _yates.load()
        mx_mask = mx.array(modmask, dtype=mx.uint64)

    per_row = max(1, len(polys) * N * 8)
    chunk = max(1, budget_bytes // per_row)

    pieces: List[np.ndarray] = []
    found = 0
    truncated = False
    for start in range(0, Z.shape[0], chunk):
        Zc = Z[start : start + chunk]
        coeffs = lift_coefficients(polys, Zc, e, L)  # (b, |I|, 2^L)
        if use_gpu:
            vals = yates.transform(mx.array(coeffs), "ZETA_SUB")
            ok = mx.all(mx.bitwise_and(vals, mx_mask) == 0, axis=1)
            ok_np = np.asarray(ok)
        else:
            vals = zeta_sub_numpy(coeffs)
            ok_np = np.all((vals & modmask) == 0, axis=1)

        bidx, sidx = np.nonzero(ok_np)
        if bidx.size:
            take = bidx.size
            if found + take > cap:
                take = cap - found
                truncated = True
            bidx, sidx = bidx[:take], sidx[:take]
            w = Zc[bidx] + (np.uint64(1 << e) * _digits(sidx, L))
            pieces.append(w)
            found += take
        if truncated:
            break

    if not pieces:
        return np.zeros((0, L), dtype=np.uint64), truncated
    return np.concatenate(pieces, axis=0), truncated


# -- one level, structural -------------------------------------------------


def _level_structural(
    polys: Sequence[Poly],
    Z: np.ndarray,
    e: int,
    L: int,
    *,
    cap: int,
) -> tuple[np.ndarray, bool]:
    """Lift every ``z`` by solving the affine F_2 system it induces.

    Valid only for ``e >= 1``, where the lift is affine in the digit vector.
    """
    if e < 1:
        raise ValueError("the structural path needs e >= 1; the lift is not affine at e = 0")
    if L > MAX_STRUCTURAL_VARS:
        raise ValueError(f"L = {L} exceeds the {MAX_STRUCTURAL_VARS}-variable vectorised path")

    B = Z.shape[0]
    P = np.uint64(1 << e)

    # Evaluate the system at z and at each z + 2^e * unit_l, in one batch.
    # coeff of d_l in the multilinear lift  ==  f(z + 2^e e_l) - f(z),
    # because setting d = 1_{l} leaves exactly the empty and {l} terms.
    pts = np.repeat(Z, L + 1, axis=0)
    within = np.tile(np.arange(L + 1), B)
    moved = within > 0
    pts[moved, within[moved] - 1] += P

    vals = eval_mod(polys, pts).reshape(B, L + 1, len(polys))
    f_z = vals[:, 0, :]  # (B, |I|)
    delta = vals[:, 1:, :] - f_z[:, None, :]  # (B, L, |I|)

    sh = np.uint64(e)
    rhs = ((f_z >> sh) & np.uint64(1)).astype(np.uint64)  # (B, |I|)
    gbits = ((delta >> sh) & np.uint64(1)).astype(np.uint64)  # (B, L, |I|)

    weights = (np.uint64(1) << np.arange(L, dtype=np.uint64))  # (L,)
    rows = (gbits * weights[None, :, None]).sum(axis=1)  # (B, |I|) bitmasks

    pieces: List[np.ndarray] = []
    found = 0
    truncated = False
    for b in range(B):
        sols = solve_affine(
            [int(v) for v in rows[b]], [int(v) for v in rhs[b]], L
        )
        if not sols.consistent:
            continue
        room = cap - found
        if sols.count > room:
            truncated = True
        masks = sols.enumerate(limit=room)
        if not masks:
            if truncated:
                break
            continue
        w = Z[b][None, :] + P * _digits(np.array(masks, dtype=np.uint64), L)
        pieces.append(w)
        found += len(masks)
        if truncated:
            break

    if not pieces:
        return np.zeros((0, L), dtype=np.uint64), truncated
    return np.concatenate(pieces, axis=0), truncated


# -- the dynamic program ---------------------------------------------------


def solve(
    polys: Sequence[Poly],
    L: int,
    max_e: int,
    *,
    backend: Backend = "structural",
    max_survivors: int = 1 << 20,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> Result:
    """Algorithm 6 at ``p = 2``.

    Args:
      polys: the system ``(f_i)_{i in I}`` over Z.
      L: number of variables.
      max_e: level bound; the loop stops here even if zeros remain.
      backend: ``"dense"`` uses the Metal ``ZETA_SUB`` at every level;
        ``"structural"`` uses it at ``e = 0`` and F_2 linear algebra above;
        ``"numpy"`` is ``"dense"`` with a CPU transform, for machines with no
        Metal device.
      max_survivors: cap on ``|Z|`` carried between levels.
      budget_bytes: coefficient-table memory per chunk.

    Returns:
      A :class:`Result`.  Read ``guarantee`` before quoting ``e_max``.
    """
    if max_e > RING_BITS:
        raise ValueError(
            f"max_e = {max_e} exceeds the {RING_BITS}-bit ring; "
            "values are exact only modulo 2^64"
        )
    if backend in ("dense", "numpy") and L > 30:
        raise ValueError(f"L = {L} needs a 2^{L}-entry table per polynomial")

    use_gpu = backend == "dense"
    Z = np.zeros((1, L), dtype=np.uint64)
    e = 0
    levels: List[LevelStat] = []
    truncated_ever = False

    while True:
        t0 = time.perf_counter()
        if backend == "structural" and e >= 1:
            name = "structural"
            W, truncated = _level_structural(polys, Z, e, L, cap=max_survivors)
        else:
            name = "dense-gpu" if use_gpu else "dense-cpu"
            W, truncated = _level_dense(
                polys, Z, e, L,
                use_gpu=use_gpu, budget_bytes=budget_bytes, cap=max_survivors,
            )
        dt = time.perf_counter() - t0
        levels.append(LevelStat(e, name, Z.shape[0], W.shape[0], truncated, dt))
        truncated_ever |= truncated

        if W.shape[0] == 0:
            reason = "W empty: no common zero modulo 2^%d" % (e + 1)
            guarantee = "lower_bound" if truncated_ever else "exact"
            if truncated_ever:
                reason += " (after discarding survivors at the cap)"
            # Z, not W: the survivors of the *previous* level are the zeros
            # modulo 2^e_max that this result is about.
            return Result(e, guarantee, reason, levels, _yates.kernel_commit(),
                          witness=[int(v) for v in Z[0]])

        e += 1
        Z = W
        if e >= max_e:
            return Result(
                e,
                "lower_bound",
                f"level bound max_e = {max_e} reached with {Z.shape[0]} zeros remaining",
                levels,
                _yates.kernel_commit(),
                witness=[int(v) for v in Z[0]],
            )
