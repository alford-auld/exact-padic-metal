"""Phase 2: the §4 reduction from a character network to a polynomial system."""

import pytest

from conftest import requires_metal
from padic import character, ddp, reference
from padic.network import NetworkInstance, planted_instance, random_instance


# -- the character as a polynomial ----------------------------------------


@pytest.mark.parametrize("p", [2, 3, 5])
@pytest.mark.parametrize("E", [1, 2, 3, 5, 8, 11])
def test_taylor_polynomial_agrees_with_algorithm_4(p, E):
    """The extracted coefficients must define the same function chi."""
    for x in range(20):
        assert character.evaluate(p, E, x) == reference.character_taylor_series(p, E, x)


@pytest.mark.parametrize("p", [2, 3, 5])
@pytest.mark.parametrize("E", [1, 2, 4, 7])
def test_character_is_a_group_homomorphism(p, E):
    """chi(x + y) = chi(x) chi(y): the defining property of a character."""
    mod = p**E
    for x in range(8):
        for y in range(8):
            lhs = character.evaluate(p, E, x + y)
            rhs = character.evaluate(p, E, x) * character.evaluate(p, E, y) % mod
            assert lhs == rhs


@pytest.mark.parametrize("p,E", [(2, 6), (3, 5)])
def test_character_equals_a_to_the_x(p, E):
    """chi = a^. for a = chi(1), computed by an unrelated algorithm."""
    a = character.character_base(p, E)
    for x in range(12):
        assert character.evaluate(p, E, x) == reference.character_binary_modular(p, E, a, x)


def test_character_is_injective_on_the_levels_it_resolves(rng):
    """Proposition 2.3: chi is injective, so it separates residues."""
    p, E = 2, 10
    seen = {character.evaluate(p, E, x) for x in range(1 << (E - 2))}
    assert len(seen) == 1 << (E - 2)


# -- the reduction ---------------------------------------------------------


def test_planted_network_is_a_zero_of_the_system(rng):
    """The network that produced Y must satisfy the reduced system exactly."""
    for _ in range(6):
        inst, z = planted_instance(rng, n_samples=3, N=2, M=1, D=1, E=3, F=1)
        mod = inst.p**inst.E_star
        for f in inst.to_polynomial_system():
            assert f.eval_int(z) % mod == 0


def test_reduction_agrees_with_the_character_itself(rng):
    """The polynomial residual and the chi-evaluated residual must match.

    ``residual_valuation`` never touches ``to_polynomial_system``, so this
    compares the reduction against an independent computation of the same
    quantity rather than against itself.
    """
    for _ in range(6):
        inst, z = planted_instance(rng, n_samples=2, N=1, M=1, D=2, E=3, F=0)
        assert inst.residual_valuation(z) == inst.E_star


@pytest.mark.parametrize("backend", ["numpy", "structural", pytest.param("dense", marks=requires_metal)])
def test_algorithm_6_finds_the_planted_network(rng, backend):
    """A fittable instance must run to the level bound, not stop early."""
    inst, _ = planted_instance(rng, n_samples=2, N=1, M=1, D=1, E=3, F=0)
    polys = inst.to_polynomial_system()
    res = ddp.solve(polys, inst.n_vars, max_e=inst.E_star, backend=backend)
    assert res.e_max == inst.E_star
    assert res.guarantee == "lower_bound"


def test_reduction_matches_the_literal_algorithm_on_unfittable_data(rng):
    """Random labels: the fast path and the pseudocode must agree on e_max."""
    for _ in range(4):
        inst = random_instance(rng, n_samples=3, N=1, M=1, D=1, E=3, F=0)
        polys = inst.to_polynomial_system()
        want = reference.digit_dynamic_programming(2, inst.n_vars, polys, max_e=inst.E_star)
        got = ddp.solve(polys, inst.n_vars, max_e=inst.E_star, backend="structural")
        assert got.e_max == want


def test_objective_valuation_divides_out_the_scaling():
    inst = NetworkInstance(((1,),), ((4,),), D=1, E=2, F=2)
    assert inst.E_star == 4
    assert inst.objective_valuation(4) == 2  # |p|^(e* - F)


def test_variable_layout_is_a_bijection():
    inst = NetworkInstance(((1, 2),), ((3,),), D=3, E=2, F=0)
    idx = (
        [inst.index_A(d, k) for d in range(inst.D) for k in range(inst.N)]
        + [inst.index_b(d) for d in range(inst.D)]
        + [inst.index_C(j, d) for j in range(inst.M) for d in range(inst.D)]
    )
    assert sorted(idx) == list(range(inst.n_vars))
    assert len(inst.variable_names()) == inst.n_vars
    assert all(inst.variable_names())
