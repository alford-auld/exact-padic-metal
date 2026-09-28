"""Literal transcriptions of the paper's pseudocode, in pure Python.

Nothing here is fast and nothing here is clever.  These functions exist so
every optimised path has a ground truth it can be checked against bit for bit,
and so the deviations this implementation makes from the printed pseudocode
are visible as code rather than as prose.  Deviations are recorded in
``docs/deviations.md``; each one also carries a comment at its site.

Source: T. Mihara, *p-adic Character Neural Network*, arXiv:2603.29905v1.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

from .poly import Poly


# -- Algorithm 1 -----------------------------------------------------------


def valuative_decomposition(p: int, x: int) -> Tuple[int, int]:
    """``ValuativeDecomposition(p, x)`` -> ``(v_p(x), x / p^v)``.

    Algorithm 1, verbatim.  ``x = 0`` would not terminate; the paper only ever
    calls this on nonzero arguments, and we make that precondition explicit.
    """
    if x == 0:
        raise ValueError("v_p(0) is undefined (the printed loop would not halt)")
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v, x


# -- Algorithm 2 -----------------------------------------------------------


def modular_factorial(p: int, E: int, N: int) -> List[Tuple[int, int]]:
    """``ModularFactorial(p, E, N)`` -> ``[(v_p(n!), unit(n!) mod p^E)]_{n<=N}``.

    Algorithm 2, verbatim.
    """
    pE = p**E
    f = [(0, 1)] * (N + 1)
    for e in range(N):
        v, u = f[e]
        v2, u2 = valuative_decomposition(p, e + 1)
        f[e + 1] = (v + v2, (u * u2) % pE)
    return f


# -- Algorithm 3 -----------------------------------------------------------


def character_mahler_series(p: int, E: int, a: int, x: int) -> int:
    """``a^x mod p^E`` by the Mahler / binomial series.  Algorithm 3, verbatim.

    ``a`` must lie in ``1 + p Z_p``.  The series is truncated at ``E`` terms
    because ``(a-1)^e`` lies in ``p^e Z_p``.
    """
    if (a - 1) % p != 0:
        raise ValueError("a must lie in 1 + p Z_p")
    pE = p**E
    y = 0
    v, u = 0, 1  # valuation and unit part of binomial(x, e)
    b = 1  # (a-1)^e
    for e in range(E):
        if e > 0:
            # binomial(x, e) = 0 once the factor (x - e + 1) vanishes, and
            # stays 0 for every larger e; u = 0 propagates exactly that.
            if x - e + 1 == 0:
                v1, u1 = 0, 0
            else:
                v1, u1 = valuative_decomposition(p, x - e + 1)
            v2, u2 = valuative_decomposition(p, e)
            # e <= E-1 here, so this modulus is at least p: the printed bound
            # p^(E-e) is safe (b already carries e factors of p).
            mod = p ** (E - e)
            v = v + v1 - v2
            u = (u * u1 * pow(u2, -1, mod)) % mod
            b = (b * (a - 1)) % pE
        if v < E:
            y = (y + p**v * u * b) % pE
    return y


# -- Algorithm 4 -----------------------------------------------------------


def character_taylor_series(p: int, E: int, x: int) -> int:
    """``exp_p(q x) mod p^E`` by the Taylor series.  Algorithm 4, *corrected*.

    DEVIATION (precision bound).  The printed line 14 sets
    ``e' = max{0, E - e m}`` and carries the running ``u`` and ``b`` only
    modulo ``p^(e')``.  That loses digits the answer depends on.  The term at
    index ``e`` is ``p^(v + m e) u b`` with ``v = -v_p(e!)``, so ``u b`` is
    needed to precision ``E - (v + m e) = E - m e + v_p(e!)``, which exceeds
    the printed ``E - m e`` whenever ``e!`` is divisible by ``p``.  Concretely
    at ``p = 2, E = 10, e = 5``: the printed bound gives ``e' = 0``, forcing
    ``u = 0``, yet ``q^5/5! = 2^10/120`` has valuation 7 and still contributes
    modulo ``2^10``.

    Nor can the loop stop when the bound reaches zero.  ``v + m e`` is
    ``m e - (e - s_p(e))/(p-1)``, which is **not** monotonic in ``e`` -- for
    ``p = 2`` it is ``e + s_2(e)``, which dips at every power of two (``e = 7``
    gives 10, ``e = 8`` gives 9).  A term can therefore fall below ``p^E``
    again after an earlier one exceeded it.

    So the running state is carried at full precision ``p^E`` and only the
    *decision to add* a term uses ``v + m e < E``.  This changes none of the
    paper's complexity claims: ``u`` and ``b`` were already elements of a
    residue ring of size at most ``p^E``.
    """
    m = 2 if p == 2 else 1
    pE = p**E
    # E' = ceil(E / (m - 1/(p-1))), the printed truncation bound (line 10).
    # Sound because v + m e >= e (m - 1/(p-1)) by Legendre's formula.
    num, den = (m * (p - 1) - 1), (p - 1)
    Ep = -((-E * den) // num)

    y = 0
    v, u = 0, 1  # valuation and unit part of (e!)^-1
    b = 1  # x^e
    for e in range(Ep):
        if e > 0:
            v1, u1 = valuative_decomposition(p, e)
            v = v - v1
            u = (u * pow(u1, -1, pE)) % pE
            b = (b * x) % pE
        if v + m * e < E:
            y = (y + p ** (v + m * e) * u * b) % pE
    return y


# -- Algorithm 5 -----------------------------------------------------------


def character_binary_modular(p: int, E: int, a: int, x: int) -> int:
    """``a^x mod p^E`` by square-and-multiply.  Algorithm 5, *corrected*.

    DEVIATION (transcription).  The printed Algorithm 5 cannot be right: it
    initialises the accumulator to ``y <- 0`` (so the result is 0 whatever the
    input), never reads its argument ``a`` at all, and squares the *exponent*
    ``x`` rather than a running base before halving it.  The three faults are
    consistent with a single conflation of the base variable and the exponent
    variable.  We implement the classical binary modular method the text names,
    which is what the surrounding complexity claim ``O((log_2 x) t(p^E))``
    describes.
    """
    pE = p**E
    y = 1
    b = a % pE
    while x:
        if x % 2 == 1:
            y = (y * b) % pE
        b = (b * b) % pE
        x //= 2
    return y


# -- Algorithm 6 -----------------------------------------------------------


def digit_dynamic_programming(
    p: int,
    L: int,
    polys: Sequence[Poly],
    max_e: int | None = None,
) -> int:
    """``DigitDynamicProgramming(p, L, f)``.  Algorithm 6, verbatim.

    Returns the largest ``e`` for which ``f`` has a common zero in
    ``Z/p^e Z``.  ``max_e`` is the paper's own remedy for the case where a
    common zero exists in ``Z_p`` and the printed ``while True`` never halts:
    "it suffices to break the while loop when ``e = E``".
    """
    e = 0
    Z = [(0,) * L]
    while True:
        W = []
        for z in Z:
            for d in itertools.product(range(p), repeat=L):
                w = tuple(zl + p**e * dl for zl, dl in zip(z, d))
                mod = p ** (e + 1)
                if all(f.eval_int(w) % mod == 0 for f in polys):
                    W.append(w)
        if not W:
            break
        e += 1
        Z = W
        if max_e is not None and e >= max_e:
            break
    return e


def brute_force_e_max(p: int, L: int, polys: Sequence[Poly], max_e: int) -> int:
    """``e_max`` by exhaustive search of ``(Z/p^e Z)^L`` at each level.

    Independent of Algorithm 6's lifting argument -- it never uses the fact
    that a zero mod ``p^(e+1)`` reduces to a zero mod ``p^e`` -- so it is a
    genuine check on the dynamic program and not a restatement of it.
    """
    best = 0
    for e in range(1, max_e + 1):
        mod = p**e
        found = any(
            all(f.eval_int(z) % mod == 0 for f in polys)
            for z in itertools.product(range(mod), repeat=L)
        )
        if not found:
            return best
        best = e
    return best
