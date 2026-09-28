"""§4: reduce a p-adic character network to a polynomial system.

The network of arXiv:2603.29905v1 §4 is

    minimise  || (y_i - C chi(A x_i + b))_{i in I} ||
    over      A in M_{D,N}(Z_p),  b in Z_p^D,  C in M_{M,D}(Q_p).

With the paper's practical hypotheses -- entries of ``X`` known modulo
``p^(E+F-1)``, entries of ``Y`` known modulo ``p^E`` and lying in
``p^-F Z_p`` -- the search space becomes finite, and multiplying through by
``p^F`` clears the only denominators.  Writing ``C' = p^F C`` and
``E* = E + F``, the residual

    f_{i,j} = p^F y_{i,j} - sum_d C'_{j,d} chi( sum_k A_{d,k} x_{i,k} + b_d )

is a **polynomial** over Z in the ``L = D N + D + M D`` unknowns, because
``chi`` truncates to a polynomial of degree ``E' - 1`` modulo ``p^E*``
(see ``character.taylor_coefficients``).  Minimising the l-infinity norm of
the residual is then exactly the feasibility question Algorithm 6 answers:
if ``e*`` is the largest level with a common zero, the minimum of the scaled
objective is ``|p|^(e*)``, hence the minimum of the original objective is
``|p|^(e* - F)``.

That the affine map preserves ``Z_p^N`` -- the whole point of using a
character rather than a ball indicator -- is what keeps ``A`` and ``b``
integral here, and so what keeps this a polynomial system at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from . import character
from .poly import Poly


@dataclass(frozen=True)
class NetworkInstance:
    """A finite, fully specified instance of the §4 optimisation problem.

    Attributes:
      X: ``(|I|, N)`` sample points, integer representatives in ``Z_p``.
      Y_scaled: ``(|I|, M)`` observations **pre-multiplied by ``p^F``**, so
        they are integers.  This is the paper's hypothesis (3), that every
        entry of ``Y`` lies in ``p^-F Z_p``, made explicit in the data type
        rather than left to a convention.
      D: hidden width.
      E: precision hyperparameter.
      F: denominator-valuation bound.
      p: the prime.
    """

    X: Tuple[Tuple[int, ...], ...]
    Y_scaled: Tuple[Tuple[int, ...], ...]
    D: int
    E: int
    F: int = 0
    p: int = 2

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def N(self) -> int:
        return len(self.X[0]) if self.X else 0

    @property
    def M(self) -> int:
        return len(self.Y_scaled[0]) if self.Y_scaled else 0

    @property
    def E_star(self) -> int:
        """``E* = E + F``: the precision the scaled system is solved at."""
        return self.E + self.F

    @property
    def n_vars(self) -> int:
        """``L = D N + D + M D``."""
        return self.D * self.N + self.D + self.M * self.D

    # -- variable layout ---------------------------------------------------
    #
    #   [ A_{d,k} ]  d*N + k                      d < D, k < N
    #   [ b_d     ]  D*N + d                      d < D
    #   [ C'_{j,d}]  D*N + D + j*D + d            j < M, d < D

    def index_A(self, d: int, k: int) -> int:
        return d * self.N + k

    def index_b(self, d: int) -> int:
        return self.D * self.N + d

    def index_C(self, j: int, d: int) -> int:
        return self.D * self.N + self.D + j * self.D + d

    def variable_names(self) -> List[str]:
        names = [""] * self.n_vars
        for d in range(self.D):
            for k in range(self.N):
                names[self.index_A(d, k)] = f"A[{d},{k}]"
            names[self.index_b(d)] = f"b[{d}]"
        for j in range(self.M):
            for d in range(self.D):
                names[self.index_C(j, d)] = f"C'[{j},{d}]"
        return names

    # -- the reduction -----------------------------------------------------

    def to_polynomial_system(self) -> List[Poly]:
        """The system ``(f_{i,j})``, one polynomial per (sample, output)."""
        p, L = self.p, self.n_vars
        mod = p**self.E_star
        coeffs = character.taylor_coefficients(p, self.E_star)

        polys: List[Poly] = []
        for i, x_i in enumerate(self.X):
            # chi(u_{i,d}) as a polynomial in A[d,:] and b[d]
            chi_at: List[Poly] = []
            for d in range(self.D):
                lin: Dict[int, int] = {self.index_A(d, k): int(x_i[k]) for k in range(self.N)}
                lin[self.index_b(d)] = 1
                coeff_vec = [0] * L
                for idx, c in lin.items():
                    coeff_vec[idx] += c
                u = Poly.linear(L, coeff_vec)
                acc = Poly.zero(L)
                power = Poly.constant(L, 1)
                for t in coeffs:
                    if t:
                        acc = acc + power * t
                    power = (power * u).reduce_mod(mod)
                chi_at.append(acc.reduce_mod(mod))

            for j in range(self.M):
                f = Poly.constant(L, int(self.Y_scaled[i][j]))
                for d in range(self.D):
                    f = f - Poly.var(L, self.index_C(j, d)) * chi_at[d]
                polys.append(f.reduce_mod(mod))
        return polys

    # -- reading an answer back -------------------------------------------

    def decode(self, z: Sequence[int]) -> Dict[str, object]:
        """Split a solution vector into ``A``, ``b`` and ``C' = p^F C``."""
        A = [[int(z[self.index_A(d, k)]) for k in range(self.N)] for d in range(self.D)]
        b = [int(z[self.index_b(d)]) for d in range(self.D)]
        C = [[int(z[self.index_C(j, d)]) for d in range(self.D)] for j in range(self.M)]
        return {"A": A, "b": b, "C_scaled": C}

    def residual_valuation(self, z: Sequence[int]) -> int:
        """``min_{i,j} v_p(scaled residual)`` evaluated through ``chi`` itself.

        Deliberately independent of :meth:`to_polynomial_system`: it calls the
        character rather than its polynomial approximation, so it can catch an
        error in the reduction instead of repeating it.  Capped at ``E*``,
        beyond which the truncated character says nothing.
        """
        p, Es = self.p, self.E_star
        mod = p**Es
        dec = self.decode(z)
        A, b, C = dec["A"], dec["b"], dec["C_scaled"]  # type: ignore[assignment]
        worst = Es
        for i, x_i in enumerate(self.X):
            chi_vals = [
                character.evaluate(p, Es, sum(A[d][k] * x_i[k] for k in range(self.N)) + b[d])
                for d in range(self.D)
            ]
            for j in range(self.M):
                r = (self.Y_scaled[i][j] - sum(C[j][d] * chi_vals[d] for d in range(self.D))) % mod
                v = Es if r == 0 else (r & -r).bit_length() - 1 if p == 2 else _val(p, r)
                worst = min(worst, v)
        return worst

    def objective_valuation(self, e_star: int) -> int:
        """Exponent of the minimum of the **original** objective.

        Algorithm 6 returns ``e*`` for the scaled system, so the scaled
        objective's minimum is ``|p|^(e*)`` and the original objective's is
        ``|p|^(e* - F)`` -- the ``p^F`` that cleared the denominators of ``C``
        is divided back out.
        """
        return e_star - self.F


def _val(p: int, x: int) -> int:
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


# -- instance generators ---------------------------------------------------


def planted_instance(
    rng: random.Random,
    *,
    n_samples: int,
    N: int,
    M: int,
    D: int,
    E: int,
    F: int = 0,
    p: int = 2,
) -> Tuple[NetworkInstance, List[int]]:
    """An instance whose labels were produced by a network, plus that network.

    The returned parameter vector is a common zero of the system modulo
    ``p^E*`` by construction, so Algorithm 6 must run to the level bound.
    This is the end-to-end check on the §4 reduction.
    """
    Es = E + F
    mod = p**Es
    A = [[rng.randrange(mod) for _ in range(N)] for _ in range(D)]
    b = [rng.randrange(mod) for _ in range(D)]
    C = [[rng.randrange(mod) for _ in range(D)] for _ in range(M)]
    X = tuple(tuple(rng.randrange(mod) for _ in range(N)) for _ in range(n_samples))

    Y = []
    for x_i in X:
        chi_vals = [
            character.evaluate(p, Es, sum(A[d][k] * x_i[k] for k in range(N)) + b[d])
            for d in range(D)
        ]
        Y.append(tuple(sum(C[j][d] * chi_vals[d] for d in range(D)) % mod for j in range(M)))

    inst = NetworkInstance(X, tuple(Y), D=D, E=E, F=F, p=p)
    z = [0] * inst.n_vars
    for d in range(D):
        for k in range(N):
            z[inst.index_A(d, k)] = A[d][k]
        z[inst.index_b(d)] = b[d]
    for j in range(M):
        for d in range(D):
            z[inst.index_C(j, d)] = C[j][d]
    return inst, z


def random_instance(
    rng: random.Random,
    *,
    n_samples: int,
    N: int,
    M: int,
    D: int,
    E: int,
    F: int = 0,
    p: int = 2,
) -> NetworkInstance:
    """Labels drawn independently of any network -- generally *not* fittable."""
    Es = E + F
    mod = p**Es
    X = tuple(tuple(rng.randrange(mod) for _ in range(N)) for _ in range(n_samples))
    Y = tuple(tuple(rng.randrange(mod) for _ in range(M)) for _ in range(n_samples))
    return NetworkInstance(X, Y, D=D, E=E, F=F, p=p)
