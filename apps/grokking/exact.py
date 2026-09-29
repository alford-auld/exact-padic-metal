"""An exact two-unit character network for modular addition.

The task from `neelnanda-io/Grokking <https://github.com/neelnanda-io/Grokking>`_
is to learn ``(a + b) mod P`` for a prime ``P`` (113 there).  A transformer
trained on it groks: it memorises, then much later generalises, and the
mechanistic account of the generalising circuit is that it computes with
**characters** of ``Z/P`` -- discrete Fourier components and trigonometric
identities.

A p-adic character network has a character as its *activation function*, so for
this task the solution is not something to be found by training.  It is two
units wide and can be written down.

Why it works
------------
``chi = exp_p(q .)`` with ``q = p^m`` (``m = 2`` at ``p = 2``, else ``m = 1``)
is a group homomorphism ``Z_p -> Q_p^x``, and its Taylor series is

    chi(v) = 1 + q v + q^2 v^2 / 2! + ...

Feed it ``v = p^j u``.  By Legendre's formula the term of index ``e`` has
valuation at least ``(j + m) e - (e - s_p(e))/(p-1)``.  The binding term is
``e = 2``, giving ``2(j + m) - v_p(2!)``, and ``v_p(2!)`` is where the two
primes differ: 1 at ``p = 2``, 0 for odd ``p``.  Dividing out the linear
coefficient ``p^(j+m)`` leaves

    (chi(p^j u) - 1) / p^(j+m)  =  u  +  O(p^((j+m) - v_p(2!)))

i.e. an error of valuation ``j + m - 1`` at ``p = 2`` and ``j + m`` for odd
``p``.  Terms with ``e >= 3`` are not binding: the same bound gives
``(e-1)[(j+m) - 1/(p-1)]``, increasing in ``e``.

Setting ``F = j + m`` and ``j = E - 1`` makes both cases exactly ``E``, so the
error vanishes modulo ``p^E``,
so with ``u = a + b`` the network

    y  =  p^-F ( chi(p^j (a+b)) - chi(0) )

computes ``(a + b) mod p^E`` **exactly**, for every prime ``p``.  In the §4
variable layout that is ``D = 2`` hidden units:

    A = [[p^j, p^j], [0, 0]]        b = [0, 0]        C' = p^F C = [1, -1]

The second unit is the constant ``chi(0) = 1``; it exists only to subtract the
1 in the Taylor series.  Nothing here is fitted -- the whole construction is
the homomorphism property plus Legendre's formula.

At the Grokking modulus
-----------------------
``P = 113`` is prime, so take ``p = 113`` and ``E = 1``: then ``j = 0``,
``F = 1``, and the solution is

    y = (chi(a + b) - 1) / 113          A = [[1,1],[0,0]],  C = [1/113, -1/113]

which reproduces all 113^2 = 12769 pairs exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from padic import character


def m_of(p: int) -> int:
    """``m`` such that ``exp_p`` converges on ``p^m Z_p``."""
    return 2 if p == 2 else 1


@dataclass(frozen=True)
class ExactSolution:
    """The closed-form parameters, in the §4 scaled convention (``C' = p^F C``)."""

    p: int
    E: int
    j: int
    F: int
    A: List[List[int]]
    b: List[int]
    C_scaled: List[List[int]]

    @property
    def E_star(self) -> int:
        return self.E + self.F

    @property
    def modulus(self) -> int:
        """The task modulus ``p^E`` this solution is exact for."""
        return self.p**self.E

    def predict_scaled(self, a: int, b: int) -> int:
        """``p^F y`` for input ``(a, b)``, modulo ``p^(E+F)``.

        Run through ``chi`` itself, not through the polynomial approximation
        the §4 reduction builds -- this is the check on that reduction, so it
        must not share code with it.

        The scaled form is the one that always makes sense: §4 puts ``C`` in
        ``M(Q_p)``, so a network's output may legitimately carry a denominator
        up to ``p^F`` and need not be a p-adic integer.  Correctness is
        ``predict_scaled(a, b) == p^F * ((a+b) mod p^E)``.
        """
        p, Es = self.p, self.E_star
        mod = p**Es
        x = (a, b)
        acc = 0
        for d in range(len(self.b)):
            u = sum(self.A[d][k] * x[k] for k in range(2)) + self.b[d]
            acc = (acc + self.C_scaled[0][d] * character.evaluate(p, Es, u)) % mod
        return acc

    def predict(self, a: int, b: int) -> int:
        """``y`` itself.  Only defined when the output has no denominator."""
        acc = self.predict_scaled(a, b)
        if acc % self.p**self.F:
            raise ValueError(
                f"output {acc} / p^{self.F} is not a p-adic integer; "
                "use predict_scaled"
            )
        return (acc // self.p**self.F) % self.modulus

    def is_correct(self, a: int, b: int) -> bool:
        pF = self.p**self.F
        return self.predict_scaled(a, b) == (pF * ((a + b) % self.modulus)) % self.p**self.E_star

    def parameter_vector(self, inst) -> List[int]:
        """The same parameters as a solution vector for ``inst``'s layout."""
        mod = self.p**self.E_star
        z = [0] * inst.n_vars
        for d in range(inst.D):
            for k in range(inst.N):
                z[inst.index_A(d, k)] = self.A[d][k] % mod
            z[inst.index_b(d)] = self.b[d] % mod
        for j in range(inst.M):
            for d in range(inst.D):
                z[inst.index_C(j, d)] = self.C_scaled[j][d] % mod
        return z


def exact_solution(p: int, E: int) -> ExactSolution:
    """The two-unit network computing ``(a + b) mod p^E``."""
    if E < 1:
        raise ValueError("E must be at least 1")
    m = m_of(p)
    j = E - 1
    F = j + m
    mod = p ** (E + F)
    return ExactSolution(
        p=p,
        E=E,
        j=j,
        F=F,
        A=[[p**j, p**j], [0, 0]],
        b=[0, 0],
        C_scaled=[[1, (-1) % mod]],
    )


def verify_all(sol: ExactSolution) -> int:
    """Mismatches over **all** ``p^E x p^E`` input pairs.  Zero means exact."""
    P = sol.modulus
    return sum(1 for a in range(P) for b in range(P) if not sol.is_correct(a, b))


def verify_sample(sol: ExactSolution, pairs: Sequence[tuple[int, int]]) -> int:
    """Mismatches over a given set of pairs."""
    return sum(1 for a, b in pairs if not sol.is_correct(a, b))


def solution_from_vector(p: int, E: int, F: int, inst, z: Sequence[int]) -> ExactSolution:
    """Wrap a parameter vector found by search in the same evaluator.

    ``j`` is not recoverable from -- and not used by -- the forward pass; it is
    recorded as ``-1`` to mark that this did not come from the closed form.
    """
    dec = inst.decode(z)
    return ExactSolution(
        p=p, E=E, j=-1, F=F,
        A=dec["A"], b=dec["b"], C_scaled=dec["C_scaled"],
    )
