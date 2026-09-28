"""Fit the Grokking task by feasibility search, and test what the fit does.

The closed form in ``exact.py`` says a two-unit exact solution *exists*.  This
module asks the different question: given only a fraction of the input pairs,
does Algorithm 6 **find** a parameter vector, and does that vector then get the
held-out pairs right?

That is the feasibility analogue of the train/test split the reference
repository trains against, with one difference worth being clear about: there
is no optimisation here and therefore no training dynamics.  Algorithm 6 either
returns a parameter vector that fits the training pairs to full precision or
proves that none exists.  So this measures **how much data pins the solution
down**, not how long a network takes to find it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from padic import ddp

from . import exact
from .task import ModularAddition, Pair


@dataclass
class FitResult:
    task: ModularAddition
    n_train: int
    n_test: int
    e_max: int
    e_star_target: int
    guarantee: str
    #: True when the search fitted the training pairs to full precision.
    fitted: bool
    train_errors: Optional[int]
    test_errors: Optional[int]
    witness: Optional[List[int]]
    seconds: float

    @property
    def generalises(self) -> bool:
        return bool(self.fitted and self.test_errors == 0)

    @property
    def test_accuracy(self) -> float:
        if self.test_errors is None or self.n_test == 0:
            return float("nan")
        return 1.0 - self.test_errors / self.n_test


def search_F(task: ModularAddition) -> int:
    """The smallest ``F`` that keeps the closed form inside the search space."""
    return exact.m_of(task.p) + task.E - 1


def fit(
    task: ModularAddition,
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    *,
    backend: str = "structural",
    max_survivors: int = 1 << 18,
) -> FitResult:
    """Run Algorithm 6 on the training pairs, then score the witness."""
    F = search_F(task)
    inst = task.instance(train_pairs, F=F)
    target = inst.E_star

    t0 = time.perf_counter()
    res = ddp.solve(
        inst.to_polynomial_system(), inst.n_vars,
        max_e=target, backend=backend, max_survivors=max_survivors,
    )
    dt = time.perf_counter() - t0

    fitted = res.e_max == target
    train_err = test_err = None
    if fitted and res.witness is not None:
        sol = exact.solution_from_vector(task.p, task.E, F, inst, res.witness)
        train_err = exact.verify_sample(sol, train_pairs)
        test_err = exact.verify_sample(sol, test_pairs)

    return FitResult(
        task=task,
        n_train=len(train_pairs),
        n_test=len(test_pairs),
        e_max=res.e_max,
        e_star_target=target,
        guarantee=res.guarantee,
        fitted=fitted,
        train_errors=train_err,
        test_errors=test_err,
        witness=res.witness,
        seconds=dt,
    )


def certify_closed_form(task: ModularAddition, pairs: Sequence[Pair]) -> Tuple[int, bool]:
    """Check the closed form against the §4 reduction it is supposed to solve.

    Returns ``(mismatches over all pairs, is a common zero of the system)``.
    The second half is the real content: it confirms that the polynomial system
    Algorithm 6 searches actually has the closed form among its zeros, so a
    failure to find it would be a search failure and not an absent solution.
    """
    sol = exact.exact_solution(task.p, task.E)
    inst = task.instance(pairs, F=sol.F)
    z = sol.parameter_vector(inst)
    mod = task.p**inst.E_star
    is_zero = all(f.eval_int(z) % mod == 0 for f in inst.to_polynomial_system())
    return exact.verify_all(sol), is_zero


def generalisation_curve(
    task: ModularAddition,
    fractions: Sequence[float],
    rng: random.Random,
    *,
    backend: str = "structural",
) -> List[FitResult]:
    """Fit at each training fraction, scored on the held-out pairs."""
    out = []
    for frac in fractions:
        train, test = task.split(random.Random(rng.randrange(1 << 30)), frac)
        out.append(fit(task, train, test, backend=backend))
    return out


# -- why the search stalls -------------------------------------------------


@dataclass
class RankProfile:
    """The mod-p Jacobian of the system, per level, in the digit DP.

    Each surviving ``z`` lifts to ``p^(L - rank)`` successors, so this rank is
    what decides whether Algorithm 6 terminates or drowns.
    """

    level: int
    n_survivors: int
    n_equations: int
    n_vars: int
    rank_min: int
    rank_max: int
    live_columns: List[str]

    @property
    def branching(self) -> int:
        return 2 ** (self.n_vars - self.rank_max)


def rank_profile(
    task: ModularAddition, pairs: Sequence[Pair], levels: int = 3, cap: int = 1 << 14
) -> List[RankProfile]:
    """Measure the branching factor of the digit DP, level by level.

    The finding this exists to record: for a system produced by the §4
    reduction, the columns of ``A`` and ``b`` are **identically zero** modulo
    ``p``.  The reason is that ``chi = exp_p(q .)`` has ``chi' = q chi`` with
    ``q = p^m``, so every derivative of the residual with respect to an entry
    of ``A`` or ``b`` carries a factor of ``p``.  Those digits therefore cannot
    be constrained at any level -- adding data does not help, because the
    constraint gradient is zero -- and the dynamic program enumerates all of
    them at every level.
    """
    import numpy as np

    from padic.ddp import _level_dense
    from padic.poly import eval_mod

    if task.p != 2:
        raise ValueError("the vectorised rank probe is written for p = 2")

    F = search_F(task)
    inst = task.instance(pairs, F=F)
    polys = inst.to_polynomial_system()
    L = inst.n_vars
    names = inst.variable_names()

    Z = np.zeros((1, L), dtype=np.uint64)
    Z, _ = _level_dense(polys, Z, 0, L, use_gpu=False, budget_bytes=1 << 28, cap=cap)

    out: List[RankProfile] = []
    for e in range(1, levels + 1):
        if Z.shape[0] == 0:
            break
        B = Z.shape[0]
        step = np.uint64(1 << e)
        pts = np.repeat(Z, L + 1, axis=0)
        within = np.tile(np.arange(L + 1), B)
        moved = within > 0
        pts[moved, within[moved] - 1] += step
        vals = eval_mod(polys, pts).reshape(B, L + 1, len(polys))
        g = ((vals[:, 1:, :] - vals[:, 0, :][:, None, :]) >> np.uint64(e)) & np.uint64(1)

        live = [names[l] for l in range(L) if g[:, l, :].any()]
        ranks = [_f2_rank(g[b].T) for b in range(min(B, 32))]
        out.append(RankProfile(e, B, len(polys), L, min(ranks), max(ranks), live))
        Z, _ = _level_dense(polys, Z, e, L, use_gpu=False, budget_bytes=1 << 28, cap=cap)
    return out


def _f2_rank(M) -> int:
    """Rank over F_2 of a small 0/1 matrix."""
    import numpy as np

    M = np.asarray(M, dtype=np.uint8).copy()
    r = 0
    for c in range(M.shape[1]):
        row = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if row is None:
            continue
        M[[r, row]] = M[[row, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r
