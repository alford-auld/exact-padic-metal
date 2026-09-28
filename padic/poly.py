"""Sparse multivariate polynomials over Z, and the multilinear lift.

The lift is the reason this module exists.  Algorithm 6 of arXiv:2603.29905v1
lifts a common zero ``z`` modulo ``p^e`` to candidates ``w = z + p^e d`` with
``d`` ranging over ``N_{<p}^L``.  For ``p = 2`` the digit vector ``d`` lies in
``{0,1}^L``, i.e. it *is* a subset of ``{0, ..., L-1}``, and then

    d_l**k == d_l    for every k >= 1, as integers,

so ``f(z + 2^e d)``, expanded in the ``d_l``, collapses to a **multilinear**
polynomial in ``d``.  Its coefficient vector, indexed by subsets, is exactly
what the subset-zeta transform turns into the table of all ``2^L`` values:

    (zeta_sub c)(S) = sum_{T subset S} c_T = f(z + 2^e 1_S).

Per monomial the collapse is an interpolation rather than a binomial sum:
``d_l`` only ever takes the values 0 and 1, so

    (z_l + P d_l)**k  ==  z_l**k  +  d_l * ((z_l + P)**k - z_l**k).

The ``d_l`` coefficient vanishes when ``k = 0``, so a monomial contributes only
to subsets of its own support -- a monomial in ``s`` variables touches ``2^s``
entries, not ``2^L``.

Arithmetic is carried in ``uint64``, where overflow is wraparound, hence exact
reduction modulo ``2^64``.  Since ``Z/2^64 ->> Z/2^(e+1)`` is a ring surjection
for every ``e + 1 <= 64``, every value this module produces is *exact* modulo
the target modulus with no headroom needed.  See ``docs/exactness.md``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

Exps = Tuple[int, ...]

#: Every intermediate is reduced here; Z/2^64 surjects onto every Z/2^(e+1).
RING_BITS = 64
RING_MASK = (1 << RING_BITS) - 1


class Poly:
    """A polynomial in ``nvars`` variables over Z, as a sparse term dict.

    Terms map an exponent tuple to an integer coefficient; zero coefficients
    are never stored, so ``bool(p)`` tests for the zero polynomial and
    ``len(p.terms)`` is the true sparsity.
    """

    __slots__ = ("nvars", "terms")

    def __init__(self, nvars: int, terms: Mapping[Exps, int] | None = None):
        self.nvars = nvars
        self.terms: Dict[Exps, int] = {}
        if terms:
            for e, c in terms.items():
                if len(e) != nvars:
                    raise ValueError(f"exponent tuple {e} has length != {nvars}")
                if c:
                    self.terms[tuple(e)] = self.terms.get(tuple(e), 0) + c
            self.terms = {e: c for e, c in self.terms.items() if c}

    # -- constructors ----------------------------------------------------

    @classmethod
    def zero(cls, nvars: int) -> "Poly":
        return cls(nvars)

    @classmethod
    def constant(cls, nvars: int, c: int) -> "Poly":
        return cls(nvars, {(0,) * nvars: c} if c else {})

    @classmethod
    def var(cls, nvars: int, i: int) -> "Poly":
        e = [0] * nvars
        e[i] = 1
        return cls(nvars, {tuple(e): 1})

    @classmethod
    def linear(cls, nvars: int, coeffs: Sequence[int], const: int = 0) -> "Poly":
        """``const + sum_i coeffs[i] * x_i``."""
        terms: Dict[Exps, int] = {}
        if const:
            terms[(0,) * nvars] = const
        for i, c in enumerate(coeffs):
            if c:
                e = [0] * nvars
                e[i] = 1
                terms[tuple(e)] = terms.get(tuple(e), 0) + c
        return cls(nvars, terms)

    # -- algebra ---------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Poly)
            and self.nvars == other.nvars
            and self.terms == other.terms
        )

    def _check(self, other: "Poly") -> None:
        if self.nvars != other.nvars:
            raise ValueError(f"arity mismatch: {self.nvars} vs {other.nvars}")

    def __add__(self, other: "Poly") -> "Poly":
        self._check(other)
        out = dict(self.terms)
        for e, c in other.terms.items():
            v = out.get(e, 0) + c
            if v:
                out[e] = v
            else:
                out.pop(e, None)
        return Poly(self.nvars, out)

    def __neg__(self) -> "Poly":
        return Poly(self.nvars, {e: -c for e, c in self.terms.items()})

    def __sub__(self, other: "Poly") -> "Poly":
        return self + (-other)

    def __mul__(self, other: "Poly | int") -> "Poly":
        if isinstance(other, int):
            return Poly(self.nvars, {e: c * other for e, c in self.terms.items()})
        self._check(other)
        out: Dict[Exps, int] = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                v = out.get(e, 0) + c1 * c2
                if v:
                    out[e] = v
                else:
                    out.pop(e, None)
        return Poly(self.nvars, out)

    __rmul__ = __mul__

    def __pow__(self, k: int) -> "Poly":
        if k < 0:
            raise ValueError("negative power")
        result = Poly.constant(self.nvars, 1)
        base = self
        while k:
            if k & 1:
                result = result * base
            base = base * base
            k >>= 1
        return result

    # -- shape -----------------------------------------------------------

    @property
    def degree(self) -> int:
        return max((sum(e) for e in self.terms), default=-1)

    def support(self) -> set[int]:
        """Indices of variables that actually occur."""
        return {i for e in self.terms for i, k in enumerate(e) if k}

    def reduce_mod(self, modulus: int) -> "Poly":
        return Poly(self.nvars, {e: c % modulus for e, c in self.terms.items()})

    def eval_int(self, z: Sequence[int]) -> int:
        """Exact evaluation over Z, in Python big integers."""
        total = 0
        for e, c in self.terms.items():
            t = c
            for zi, ki in zip(z, e):
                if ki:
                    t *= zi**ki
            total += t
        return total

    def multilinear_mod2(self) -> "Poly":
        """Reduce by ``x_l^2 = x_l`` and by 2: the induced function on F_2^L.

        Over F_2 every element is idempotent, so a polynomial and its
        multilinear reduction define the same function on ``{0,1}^L``.
        """
        out: Dict[Exps, int] = {}
        for e, c in self.terms.items():
            if c % 2 == 0:
                continue
            key = tuple(1 if k else 0 for k in e)
            out[key] = out.get(key, 0) ^ 1
        return Poly(self.nvars, {e: c for e, c in out.items() if c})


# -- the multilinear lift --------------------------------------------------


def _pow_u64(base: np.ndarray, k: int) -> np.ndarray:
    """``base ** k`` in uint64 (wraparound = exact mod 2^64)."""
    result = np.ones_like(base)
    b = base
    while k:
        if k & 1:
            result = result * b
        b = b * b
        k >>= 1
    return result


def _subsets(items: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    """All subsets of ``items``, as tuples, smallest first."""
    n = len(items)
    for mask in range(1 << n):
        yield tuple(items[i] for i in range(n) if (mask >> i) & 1)


def lift_coefficients(
    polys: Sequence[Poly],
    Z: np.ndarray,
    e: int,
    nvars: int,
) -> np.ndarray:
    """Multilinear coefficients of ``f_i(z + 2^e d)`` as a function of ``d``.

    Args:
      polys: the system ``(f_i)_{i in I}``.
      Z: ``(B, nvars)`` uint64 array of base points, one row per surviving ``z``.
      e: current level; the digit block is scaled by ``P = 2^e``.
      nvars: ``L``.

    Returns:
      ``(B, |I|, 2^L)`` uint64 array ``c`` with
      ``c[b, i, S] = [coefficient of prod_{l in S} d_l in f_i(Z[b] + 2^e d)]``,
      exact modulo ``2^64``.  Feed it to ``ZETA_SUB`` to get the value table.

    This is dense in its output by construction -- the caller wants a
    ``2^L``-point array to transform -- but the *work* is proportional to
    ``sum_terms 2^{|support|}``, not to ``|I| * 2^L``.
    """
    if nvars > 30:
        raise ValueError(f"L = {nvars} would need a 2^{nvars}-entry table")
    B = Z.shape[0]
    N = 1 << nvars
    P = np.uint64(1 << e) if e < 64 else np.uint64(0)

    out = np.zeros((B, len(polys), N), dtype=np.uint64)
    # z_shift[:, l] = z_l + 2^e, the value of the l-th coordinate when d_l = 1
    z_shift = Z + P

    for i, f in enumerate(polys):
        for exps, coeff in f.terms.items():
            c0 = np.uint64(coeff & RING_MASK)
            supp = [l for l, k in enumerate(exps) if k]
            # per-variable pair (value at d_l = 0, increment when d_l = 1)
            const: Dict[int, np.ndarray] = {}
            delta: Dict[int, np.ndarray] = {}
            for l in supp:
                k = exps[l]
                a0 = _pow_u64(Z[:, l], k)
                a1 = _pow_u64(z_shift[:, l], k)
                const[l] = a0
                delta[l] = a1 - a0
            for S in _subsets(supp):
                mask = 0
                acc = np.full(B, c0, dtype=np.uint64)
                for l in supp:
                    if l in S:
                        mask |= 1 << l
                        acc = acc * delta[l]
                    else:
                        acc = acc * const[l]
                out[:, i, mask] += acc
    return out


def eval_mod(
    polys: Sequence[Poly],
    Z: np.ndarray,
) -> np.ndarray:
    """``(B, |I|)`` uint64 table of ``f_i(z)``, exact modulo ``2^64``."""
    B = Z.shape[0]
    out = np.zeros((B, len(polys)), dtype=np.uint64)
    for i, f in enumerate(polys):
        acc = np.zeros(B, dtype=np.uint64)
        for exps, coeff in f.terms.items():
            t = np.full(B, np.uint64(coeff & RING_MASK), dtype=np.uint64)
            for l, k in enumerate(exps):
                if k:
                    t = t * _pow_u64(Z[:, l], k)
            acc = acc + t
        out[:, i] = acc
    return out
