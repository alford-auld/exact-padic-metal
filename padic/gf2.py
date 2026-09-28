"""Affine systems over F_2, as row bitmasks.

At every level ``e >= 1`` the surviving digit vectors of Algorithm 6 form the
solution set of an affine system over F_2 (see ``ddp.py``), which is either
empty or a coset of a linear subspace.  This module solves such a system and
enumerates the coset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class AffineSolutionSet:
    """Solutions of ``G d = c`` over F_2, as ``particular + span(basis)``.

    Vectors are little-endian bitmasks: bit ``l`` is coordinate ``d_l``.
    ``count`` is the exact number of solutions even when ``basis`` is large.
    """

    nvars: int
    consistent: bool
    particular: int
    basis: Tuple[int, ...]

    @property
    def count(self) -> int:
        return (1 << len(self.basis)) if self.consistent else 0

    def enumerate(self, limit: int | None = None) -> List[int]:
        """Solution bitmasks, at most ``limit`` of them (Gray-code order).

        Gray code means one basis vector is toggled per step, so building the
        list costs one XOR per solution rather than one per (solution, basis)
        pair.
        """
        if not self.consistent:
            return []
        total = self.count
        n = total if limit is None else min(total, limit)
        out = [0] * n
        cur = self.particular
        prev_gray = 0
        for i in range(n):
            gray = i ^ (i >> 1)
            delta = gray ^ prev_gray
            if delta:
                cur ^= self.basis[delta.bit_length() - 1]
                prev_gray = gray
            out[i] = cur
        return out


def solve_affine(rows: Sequence[int], rhs: Sequence[int], nvars: int) -> AffineSolutionSet:
    """Solve ``G d = c`` over F_2.

    Args:
      rows: one bitmask per equation; bit ``l`` set means ``d_l`` occurs.
      rhs: the right-hand side bit of each equation.
      nvars: number of unknowns ``L``.
    """
    work = [(int(r), int(b) & 1) for r, b in zip(rows, rhs)]
    pivot_of_col: dict[int, int] = {}  # column -> index into `reduced`
    reduced: List[Tuple[int, int]] = []

    for mask, b in work:
        for col, idx in pivot_of_col.items():
            if (mask >> col) & 1:
                pmask, pb = reduced[idx]
                mask ^= pmask
                b ^= pb
        if mask == 0:
            if b:
                return AffineSolutionSet(nvars, False, 0, ())
            continue
        col = (mask & -mask).bit_length() - 1
        # back-substitute the new pivot into the rows already stored, so the
        # system stays in reduced row echelon form
        for j, (m2, b2) in enumerate(reduced):
            if (m2 >> col) & 1:
                reduced[j] = (m2 ^ mask, b2 ^ b)
        pivot_of_col[col] = len(reduced)
        reduced.append((mask, b))

    pivots = sorted(pivot_of_col)
    free = [c for c in range(nvars) if c not in pivot_of_col]

    # particular solution: all free variables zero
    particular = 0
    for col in pivots:
        _, b = reduced[pivot_of_col[col]]
        if b:
            particular |= 1 << col
    # nullspace basis: one vector per free column
    basis = []
    for fc in free:
        vec = 1 << fc
        for col in pivots:
            m, _ = reduced[pivot_of_col[col]]
            if (m >> fc) & 1:
                vec |= 1 << col
        basis.append(vec)

    return AffineSolutionSet(nvars, True, particular, tuple(basis))
