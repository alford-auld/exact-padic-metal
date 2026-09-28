"""The F_2 solver, and the claim that it finds the same survivors as the transform."""

import itertools
import random

import numpy as np
import pytest

from conftest import random_system
from padic.ddp import _level_dense, _level_structural
from padic.gf2 import solve_affine
from padic.poly import Poly


def brute_solutions(rows, rhs, L):
    out = []
    for d in range(1 << L):
        if all(bin(r & d).count("1") % 2 == b for r, b in zip(rows, rhs)):
            out.append(d)
    return out


@pytest.mark.parametrize("L", [1, 2, 3, 4, 5])
def test_solve_affine_matches_exhaustive_search(L):
    rng = random.Random(1234 + L)
    for _ in range(80):
        neq = rng.randint(1, L + 2)
        rows = [rng.randrange(1 << L) for _ in range(neq)]
        rhs = [rng.randrange(2) for _ in range(neq)]
        want = brute_solutions(rows, rhs, L)
        sol = solve_affine(rows, rhs, L)
        assert sol.count == len(want)
        assert sorted(sol.enumerate()) == want


def test_enumerate_respects_the_limit():
    sol = solve_affine([0], [0], 6)  # 0 = 0: every vector solves it
    assert sol.count == 64
    assert len(sol.enumerate(limit=10)) == 10
    assert len(set(sol.enumerate(limit=10))) == 10


def test_inconsistent_system_has_no_solutions():
    sol = solve_affine([0b011, 0b011], [0, 1], 3)
    assert not sol.consistent
    assert sol.count == 0
    assert sol.enumerate() == []


# -- the two level implementations must agree ------------------------------


@pytest.mark.parametrize("e", [1, 2, 3])
@pytest.mark.parametrize("L", [2, 3, 4])
def test_dense_and_structural_find_the_same_survivor_set(rng, e, L):
    """Same set, not merely the same count -- the orderings differ."""
    # Plant a common zero so the dynamic program actually reaches level e;
    # an unconstrained random system usually dies at e = 0 or e = 1.
    z0 = [rng.randrange(1 << (e + 1)) for _ in range(L)]
    polys = [
        f - Poly.constant(L, f.eval_int(z0)) for f in random_system(rng, L, 2)
    ]
    # Z must consist of genuine zeros mod 2^e for the structural path's
    # premise (f_i(z) = 2^e c_i) to hold, so build it with the dense path.
    Z = np.zeros((1, L), dtype=np.uint64)
    for level in range(e):
        Z, _ = _level_dense(polys, Z, level, L, use_gpu=False,
                            budget_bytes=1 << 26, cap=1 << 16)
        if Z.shape[0] == 0:
            pytest.skip("system dies before reaching the requested level")

    dense, _ = _level_dense(polys, Z, e, L, use_gpu=False,
                            budget_bytes=1 << 26, cap=1 << 16)
    struct, _ = _level_structural(polys, Z, e, L, cap=1 << 16)

    as_set = lambda a: {tuple(int(v) for v in row) for row in a}
    assert as_set(dense) == as_set(struct)
    assert dense.shape[0] == struct.shape[0]  # no duplicates either
