"""Every optimised path must reproduce the literal pseudocode, exactly."""

import numpy as np
import pytest

from conftest import random_system, requires_metal
from padic import ddp, reference
from padic.poly import Poly

BACKENDS = ["numpy", "structural", pytest.param("dense", marks=requires_metal)]


def plant_zero(polys, z0):
    """Shift each f_i so that z0 is a common zero over Z (hence in Z_2)."""
    return [f - Poly.constant(f.nvars, f.eval_int(z0)) for f in polys]


# -- the reference itself --------------------------------------------------


@pytest.mark.parametrize("L", [1, 2, 3])
def test_reference_matches_exhaustive_search(rng, L):
    """Algorithm 6 vs. brute force over (Z/2^e)^L, which never uses lifting."""
    max_e = 4
    for _ in range(12):
        polys = random_system(rng, L, 2)
        assert reference.digit_dynamic_programming(2, L, polys, max_e=max_e) == \
            reference.brute_force_e_max(2, L, polys, max_e)


def test_reference_handles_odd_prime(rng):
    """Algorithm 6 is written for general p; only our fast paths assume p = 2."""
    L = 2
    for _ in range(8):
        polys = random_system(rng, L, 2)
        assert reference.digit_dynamic_programming(3, L, polys, max_e=3) == \
            reference.brute_force_e_max(3, L, polys, 3)


# -- the fast paths --------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_backend_matches_reference_on_random_systems(rng, backend, L):
    max_e = 6
    for _ in range(15):
        polys = random_system(rng, L, 3)
        want = reference.digit_dynamic_programming(2, L, polys, max_e=max_e)
        got = ddp.solve(polys, L, max_e=max_e, backend=backend)
        assert got.e_max == want, f"{backend}: {got.e_max} != {want}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_matches_reference_with_a_planted_zero(rng, backend):
    """A system with a zero in Z_2 runs to the level bound, not to an answer."""
    L, max_e = 3, 5
    for _ in range(10):
        z0 = [rng.randrange(8) for _ in range(L)]
        polys = plant_zero(random_system(rng, L, 3), z0)
        got = ddp.solve(polys, L, max_e=max_e, backend=backend)
        assert got.e_max == max_e
        assert got.guarantee == "lower_bound"
        assert "level bound" in got.reason


@pytest.mark.parametrize("backend", BACKENDS)
def test_survivor_sets_agree_level_by_level(rng, backend):
    """Not just e_max: the *set* Z must match the reference at every level."""
    L, max_e = 3, 4
    for _ in range(8):
        polys = random_system(rng, L, 2)
        want_levels = _reference_levels(2, L, polys, max_e)
        got = ddp.solve(polys, L, max_e=max_e, backend=backend)
        assert [s.survivors_out for s in got.levels] == want_levels


def _reference_levels(p, L, polys, max_e):
    """Sizes of W at each level of the literal algorithm."""
    import itertools

    sizes = []
    e, Z = 0, [(0,) * L]
    while True:
        mod = p ** (e + 1)
        W = [
            w
            for z in Z
            for d in itertools.product(range(p), repeat=L)
            if all(f.eval_int(w := tuple(zl + p**e * dl for zl, dl in zip(z, d))) % mod == 0
                   for f in polys)
        ]
        sizes.append(len(W))
        if not W:
            return sizes
        e, Z = e + 1, W
        if e >= max_e:
            return sizes


# -- guarantees ------------------------------------------------------------


def test_exact_when_the_program_terminates_on_its_own(rng):
    polys = [Poly(1, {(0,): 1})]  # f = 1, no zero even mod 2
    got = ddp.solve(polys, 1, max_e=8, backend="numpy")
    assert got.e_max == 0
    assert got.guarantee == "exact"


def test_cap_downgrades_the_guarantee_to_a_lower_bound(rng):
    """Discarding survivors can only under-report e_max, never over-report."""
    L, max_e = 4, 5
    z0 = [rng.randrange(4) for _ in range(L)]
    polys = plant_zero(random_system(rng, L, 1), z0)
    full = ddp.solve(polys, L, max_e=max_e, backend="numpy")
    capped = ddp.solve(polys, L, max_e=max_e, backend="numpy", max_survivors=2)
    assert capped.guarantee == "lower_bound"
    assert capped.e_max <= full.e_max


def test_max_e_beyond_the_ring_is_refused():
    with pytest.raises(ValueError, match="64-bit ring"):
        ddp.solve([Poly(1, {(1,): 1})], 1, max_e=65)


def test_structural_path_refuses_level_zero():
    with pytest.raises(ValueError, match="e >= 1"):
        ddp._level_structural([Poly(1, {(1,): 1})], np.zeros((1, 1), np.uint64), 0, 1, cap=10)
