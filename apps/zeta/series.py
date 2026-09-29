"""Poincare series coefficients, and the rational function behind them.

For a system ``f`` over Z, the Poincare series of ``f`` at ``p`` is

    P(t) = sum_{e >= 0} N_e t^e ,    N_e = #{ x in (Z/p^e Z)^L : f(x) = 0 } .

The survivor counts of Algorithm 6 *are* the ``N_e``: every common zero
modulo ``p^(e+1)`` reduces to one modulo ``p^e``, so the digit dynamic program
enumerates exactly the zero set at each level.  What the network application
had to treat as a blow-up is, here, the answer.

Igusa proved ``P(t)`` is a rational function, so a handful of terms determines
the whole series -- which is why the survivor cap that made the §4 search
hopeless barely bites here.  :func:`fit_rational` recovers it from the terms.

Everything is exact: counts are Python integers, the recurrence is solved over
``Fraction``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

import numpy as np

from padic.ddp import _level_dense, _level_structural
from padic.poly import Poly


@dataclass
class Rational:
    """``P(t) / Q(t)`` with ``Q(0) = 1``."""

    num: List[Fraction]
    den: List[Fraction]

    def __str__(self) -> str:
        return f"({_poly_str(self.num)}) / ({_poly_str(self.den)})"

    def coefficients(self, upto: int) -> List[Fraction]:
        """Expand the series back out, as a check on the fit."""
        out: List[Fraction] = []
        for n in range(upto + 1):
            v = self.num[n] if n < len(self.num) else Fraction(0)
            for i in range(1, min(n, len(self.den) - 1) + 1):
                v -= self.den[i] * out[n - i]
            out.append(v / self.den[0])
        return out


@dataclass
class PoincareResult:
    N: List[int]
    #: Largest index unaffected by the survivor cap; beyond it the counts are
    #: lower bounds, not counts, and must not be fitted.
    exact_upto: int
    seconds: float
    backend: str
    rational: Optional[Rational] = None
    recurrence: Optional[List[Fraction]] = None
    levels: List[tuple] = field(default_factory=list)

    @property
    def ratios(self) -> List[float]:
        return [self.N[i + 1] / self.N[i]
                for i in range(len(self.N) - 1) if self.N[i]]

    @property
    def dimension_estimate(self) -> Optional[float]:
        """``log_2`` of the eventual growth rate of ``N_e``.

        Taken from the dominant root of the recurrence's characteristic
        polynomial when one was found -- that is exact.  A sliding geometric
        mean is not: these series are frequently periodic (``x^2`` alternates
        ratios 1, 2, 1, 2), so a short window reads off the phase rather than
        the growth.  The fallback averages over six ratios for that reason.
        """
        import math

        if self.recurrence:
            char = [1.0] + [float(-c) for c in self.recurrence]
            roots = np.roots(char)
            if len(roots):
                top = max(abs(r) for r in roots)
                return math.log2(top) if top > 0 else None
        r = self.ratios[-6:]
        if not r:
            return None
        g = math.prod(r) ** (1 / len(r))
        return math.log2(g) if g > 0 else None


def poincare(
    polys: Sequence[Poly],
    L: int,
    upto: int = 8,
    *,
    backend: str = "structural",
    max_survivors: int = 1 << 22,
    budget_bytes: int = 1 << 28,
) -> PoincareResult:
    """Compute ``N_0 .. N_upto`` by running the digit dynamic program.

    ``N_0 = 1`` always: ``(Z/p^0 Z)^L`` is a single point and every system
    vanishes on it.
    """
    t0 = time.perf_counter()
    N = [1]
    Z = np.zeros((1, L), dtype=np.uint64)
    exact_upto = upto
    levels = []
    use_gpu = backend == "dense"

    for e in range(upto):
        if Z.shape[0] == 0:
            N.append(0)
            levels.append((e, 0, False))
            continue
        if backend != "dense" and e >= 1:
            Z, trunc = _level_structural(polys, Z, e, L, cap=max_survivors)
            name = "structural"
        else:
            Z, trunc = _level_dense(polys, Z, e, L, use_gpu=use_gpu,
                                    budget_bytes=budget_bytes, cap=max_survivors)
            name = "dense-gpu" if use_gpu else "dense-cpu"
        if trunc and exact_upto == upto:
            exact_upto = e  # N_{e+1} onwards are lower bounds
        N.append(int(Z.shape[0]))
        levels.append((e, int(Z.shape[0]), trunc))

    res = PoincareResult(N, exact_upto, time.perf_counter() - t0, name, levels=levels)
    fit = fit_rational(N[: exact_upto + 1])
    if fit is not None:
        res.recurrence, res.rational = fit
    return res


# -- recovering the rational function --------------------------------------


def find_recurrence(a: Sequence[int], slack: int = 2) -> Optional[List[Fraction]]:
    """Smallest ``c`` with ``a_n = sum_i c_i a_{n-i}`` for every available ``n``.

    ``slack`` extra equations beyond the unknowns are required, so a
    recurrence is only reported when it is over-determined and still holds --
    otherwise any ``d`` terms can be "explained" by a ``d``-term recurrence.
    """
    n = len(a)
    for d in range(1, (n - slack) // 2 + 1):
        rows = [[Fraction(a[m - i]) for i in range(1, d + 1)] for m in range(d, n)]
        rhs = [Fraction(a[m]) for m in range(d, n)]
        if len(rows) < d + slack:
            continue
        sol = _solve_exact(rows, rhs, d)
        if sol is not None:
            return sol
    return None


def _solve_exact(rows: List[List[Fraction]], rhs: List[Fraction], nvars: int):
    """Least-structure exact solve: returns a solution iff one fits *every* row."""
    M = [r[:] + [b] for r, b in zip(rows, rhs)]
    piv_of_col = {}
    r = 0
    for c in range(nvars):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = M[r][c]
        M[r] = [v / inv for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [v - f * w for v, w in zip(M[i], M[r])]
        piv_of_col[c] = r
        r += 1
    for i in range(r, len(M)):
        if M[i][nvars] != 0:
            return None  # inconsistent: no recurrence of this order
    if len(piv_of_col) < nvars:
        return None  # under-determined: not enough evidence, try a smaller d
    return [M[piv_of_col[c]][nvars] for c in range(nvars)]


def fit_rational(a: Sequence[int], slack: int = 2):
    """``(recurrence, P(t)/Q(t))`` reproducing ``a``, or ``None``."""
    c = find_recurrence(a, slack)
    if c is None:
        return None
    d = len(c)
    den = [Fraction(1)] + [-ci for ci in c]
    num = []
    for n in range(d):
        v = Fraction(a[n])
        for i in range(1, min(n, d) + 1):
            v -= c[i - 1] * a[n - i]
        num.append(v)
    while len(num) > 1 and num[-1] == 0:
        num.pop()
    rat = Rational(num, den)
    if [Fraction(x) for x in a] != rat.coefficients(len(a) - 1):
        return None  # the fit must reproduce every term it was given
    return c, rat


def _poly_str(coeffs: Sequence[Fraction], var: str = "t") -> str:
    parts = []
    for n, c in enumerate(coeffs):
        if c == 0:
            continue
        mono = "" if n == 0 else (var if n == 1 else f"{var}^{n}")
        if mono and abs(c) == 1:
            parts.append(("- " if c < 0 else "+ ") + mono)
        else:
            parts.append(("- " if c < 0 else "+ ") + f"{abs(c)}" + (f"*{mono}" if mono else ""))
    if not parts:
        return "0"
    s = " ".join(parts)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]
