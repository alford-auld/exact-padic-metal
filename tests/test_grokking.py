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


def test_grokking_constant_is_the_reference_task():
    assert GROKKING.p == 113 and GROKKING.E == 1
    assert GROKKING.modulus == 113
    assert GROKKING.n_pairs == 12769
