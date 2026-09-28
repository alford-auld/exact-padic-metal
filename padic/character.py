"""The p-adic character chi, and its polynomial approximation.

The paper offers three ways to evaluate a character modulo ``p^E`` (§2):
the Mahler series (Algorithm 3), the Taylor series (Algorithm 4), and the
binary modular method (Algorithm 5).  All three are transcribed in
``reference.py``.  Only the Taylor method is useful for the reduction in §4,
for the reason the paper gives: "it gives information of coefficients of
polynomial approximation".  Those coefficients are what this module extracts.

For ``chi = exp_p(q .)`` with ``q = 4`` at ``p = 2`` and ``q = p`` otherwise,

    chi(u) = sum_{e >= 0} (q^e / e!) u^e,

and ``q^e / e!`` lies in ``p^ceil((m - 1/(p-1)) e) Z_p``, so the series
truncates after ``E' = ceil(E / (m - 1/(p-1)))`` terms modulo ``p^E``.
``taylor_coefficients`` returns those ``E'`` coefficients as integers modulo
``p^E``, turning ``chi`` into an honest polynomial -- which is exactly what
lets §4 express the network residual as a polynomial system.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from .reference import valuative_decomposition


def truncation_order(p: int, E: int) -> int:
    """``E'``: the number of Taylor terms that survive modulo ``p^E``."""
    m = 2 if p == 2 else 1
    num, den = (m * (p - 1) - 1), (p - 1)
    return -((-E * den) // num)


@lru_cache(maxsize=None)
def taylor_coefficients(p: int, E: int) -> Tuple[int, ...]:
    """``[q^e / e! mod p^E]`` for ``e < E'``.

    Exact: ``q^e / e!`` is a p-adic integer, so its residue modulo ``p^E`` is
    well defined, and it is computed from the valuation/unit split rather than
    from a rational division.
    """
    m = 2 if p == 2 else 1
    pE = p**E
    out: List[int] = []
    v, u = 0, 1  # valuation and unit part of (e!)^-1
    for e in range(truncation_order(p, E)):
        if e > 0:
            v1, u1 = valuative_decomposition(p, e)
            v -= v1
            u = (u * pow(u1, -1, pE)) % pE
        out.append((p ** (v + m * e) * u) % pE if v + m * e < E else 0)
    return tuple(out)


def character_base(p: int, E: int) -> int:
    """``a = exp_p(q) mod p^E``, the value that determines ``chi = a^.``."""
    return sum(taylor_coefficients(p, E)) % p**E


def evaluate(p: int, E: int, x: int) -> int:
    """``chi(x) mod p^E`` from the truncated polynomial."""
    pE = p**E
    y, b = 0, 1
    for t in taylor_coefficients(p, E):
        y = (y + t * b) % pE
        b = (b * x) % pE
    return y
