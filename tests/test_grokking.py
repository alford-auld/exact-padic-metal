"""The Grokking application: the closed form, and the obstruction to finding it."""

import random

import pytest

from apps.grokking import certify, exact
from apps.grokking.task import GROKKING, ModularAddition


@pytest.mark.parametrize("p,E", [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (5, 2), (7, 1)])
def test_closed_form_is_exact_on_every_pair(p, E):
    sol = exact.exact_solution(p, E)
    assert exact.verify_all(sol) == 0


def test_closed_form_is_exact_at_the_grokking_modulus():
    """P = 113, all 12769 pairs, two hidden units."""
    sol = exact.exact_solution(113, 1)
    assert sol.modulus == 113
    assert sol.F == 1 and sol.j == 0
    assert exact.verify_all(sol) == 0


@pytest.mark.parametrize("p,E", [(2, 1), (2, 2), (2, 3), (3, 2), (113, 1)])
def test_closed_form_solves_the_reduced_system(p, E):
    """The search space really does contain it, so a miss is a search failure."""
    task = ModularAddition(p, E)
    pairs = task.all_pairs()
    probe = pairs if len(pairs) <= 64 else random.Random(0).sample(pairs, 30)
    mismatches, is_zero = certify.certify_closed_form(task, probe)
    assert mismatches == 0
    assert is_zero


def test_second_unit_is_only_the_constant():
    """Unit 1 is chi(0) = 1; it exists to subtract the Taylor series' leading 1."""
    sol = exact.exact_solution(2, 3)
    assert sol.A[1] == [0, 0] and sol.b[1] == 0
    assert sol.C_scaled[0][0] == 1
    assert sol.C_scaled[0][1] == (-1) % 2**sol.E_star


@pytest.mark.parametrize("E", [1, 2])
def test_search_recovers_an_exact_fit(E):
    """Algorithm 6 finds a solution and it gets every held-out pair right."""
    task = ModularAddition(2, E)
    train, test = task.split(random.Random(11), 1.0)
    r = certify.fit(task, train, test, max_survivors=1 << 16)
    assert r.fitted
    assert r.e_max == r.e_star_target
    assert r.train_errors == 0


def test_search_generalises_from_half_the_pairs():
    task = ModularAddition(2, 2)
    train, test = task.split(random.Random(11), 0.5)
    r = certify.fit(task, train, test, max_survivors=1 << 16)
    assert r.fitted and r.train_errors == 0
    assert r.test_errors == 0, "fit on half the pairs should still be exact"


def test_A_and_b_are_invisible_modulo_p():
    """The obstruction, as a test.

    chi' = q chi with q = p^m, so every derivative of the residual in an entry
    of A or b carries a factor of p.  Those columns of the mod-2 Jacobian are
    identically zero at every level, which is why the digit DP branches by
    2^(L - 1) per survivor no matter how much data it is given.
    """
    task = ModularAddition(2, 3)
    profiles = certify.rank_profile(task, task.all_pairs(), levels=2)
    assert profiles
    for rp in profiles:
        assert rp.n_equations == 64, "heavily over-determined, and it does not help"
        assert rp.rank_max == 1
        assert all(name.startswith("C'") for name in rp.live_columns)
        assert rp.branching == 2 ** (rp.n_vars - 1)


@pytest.mark.parametrize("N,M,D,nI", [
    (1, 1, 1, 8), (1, 1, 4, 8), (2, 1, 2, 16), (2, 1, 2, 64), (2, 1, 3, 64),
    (4, 1, 3, 128), (1, 2, 2, 16), (2, 2, 2, 64), (3, 2, 2, 64),
    (2, 3, 3, 64), (3, 4, 2, 64),
])
def test_plain_jacobian_rank_equals_M_exactly(N, M, D, nI):
    """rank_{F_p} J = M, independent of D, N and the sample count.

    chi lands in 1 + q Z_p, so every entry of every live C' column is 1 mod p
    and each output row contributes rank exactly one; A and b columns vanish.
    This is the bound the README states, so it is checked rather than quoted.
    """
    from padic.network import planted_instance
    from apps.grokking.certify import _f2_rank

    inst, z_star = planted_instance(random.Random(11), n_samples=nI,
                                    N=N, M=M, D=D, E=3, F=2)
    polys = inst.to_polynomial_system()
    L, names = inst.n_vars, inst.variable_names()
    for t in (1, 2, 3):
        z = [v % (1 << t) for v in z_star]
        base = [f.eval_int(z) for f in polys]
        cols = []
        for l in range(L):
            zz = list(z)
            zz[l] += 1 << t
            cols.append([((f.eval_int(zz) - b) >> t) & 1
                         for f, b in zip(polys, base)])
        assert _f2_rank([list(r) for r in zip(*cols)]) == M, f"t={t}"
        live = {names[l] for l in range(L) if any(cols[l])}
        assert all(n.startswith("C'") for n in live), f"t={t}: {live}"


def test_grokking_constant_is_the_reference_task():
    assert GROKKING.p == 113 and GROKKING.E == 1
    assert GROKKING.modulus == 113
    assert GROKKING.n_pairs == 12769
