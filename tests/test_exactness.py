"""The exactness contract: why wraparound at 2^64 costs nothing here.

Algorithm 6 only ever asks whether a value vanishes modulo ``2^(e+1)``.  The
kernel computes modulo ``2^64``.  Since ``Z/2^64 ->> Z/2^(e+1)`` is a ring
surjection for every ``e + 1 <= 64``, the reduction of the kernel's answer *is*
the answer -- overflow is not an approximation to be bounded, it is the
correct arithmetic in a quotient we then project further.  ``ZETA_SUB`` is
unipotent, so it loses no bits either.  These tests hold both halves down.
"""

import numpy as np
import pytest

from conftest import random_system, requires_metal
from padic import _yates
from padic.ddp import zeta_sub_numpy
from padic.poly import Poly, lift_coefficients


@requires_metal
def test_kernel_zeta_matches_the_cpu_transform_bitwise():
    import mlx.core as mx

    yates = _yates.load()
    rng = np.random.default_rng(7)
    for n in range(1, 12):
        a = rng.integers(0, 1 << 63, size=(3, 2, 1 << n), dtype=np.uint64)
        got = np.asarray(yates.transform(mx.array(a), "ZETA_SUB"))
        assert np.array_equal(got, zeta_sub_numpy(a))


@requires_metal
def test_zeta_sub_loses_no_bits():
    """Unipotent generator: the transform is an automorphism of Z/2^64."""
    yates = _yates.load()
    for n in range(1, 20):
        assert yates.bit_loss("ZETA_SUB", n) == 0
        assert yates.guaranteed_bits("ZETA_SUB", n, 64) == 64


@requires_metal
def test_mobius_inverts_zeta_bitwise():
    import mlx.core as mx

    yates = _yates.load()
    rng = np.random.default_rng(11)
    a = mx.array(rng.integers(0, 1 << 63, size=(4, 1 << 10), dtype=np.uint64))
    assert mx.array_equal(yates.mobius_sub(yates.zeta_sub(a)), a).item()


def test_overflowing_coefficients_still_give_exact_residues(rng):
    """Coefficients far past 2^64 must not perturb the answer mod 2^(e+1).

    The polynomial is built with deliberately huge integer coefficients, so
    every intermediate wraps; the value table must still agree with exact
    Python big-integer arithmetic at the modulus Algorithm 6 tests.
    """
    L, e = 3, 4
    huge = 1 << 200
    polys = [
        Poly(L, {(2, 1, 0): huge + 3, (0, 0, 3): -huge * 7 + 5, (0, 0, 0): huge // 3}),
        Poly(L, {(1, 1, 1): huge * 11 + 1, (1, 0, 0): -(huge + 9)}),
    ]
    Z = np.array([[rng.randrange(1 << e) for _ in range(L)] for _ in range(4)],
                 dtype=np.uint64)
    table = zeta_sub_numpy(lift_coefficients(polys, Z, e, L))
    mod = 1 << (e + 1)
    for b in range(Z.shape[0]):
        for S in range(1 << L):
            w = [int(Z[b, l]) + (1 << e) * ((S >> l) & 1) for l in range(L)]
            for i, f in enumerate(polys):
                assert int(table[b, i, S]) % mod == f.eval_int(w) % mod


def test_top_bit_inputs_are_not_treated_as_signed(rng):
    """Values above 2^63 must survive; a signed path would be UB or negative."""
    L, e = 2, 1
    polys = [Poly(L, {(1, 0): (1 << 63) + 1, (0, 1): (1 << 63) - 1})]
    Z = np.array([[1, 1], [0, 3]], dtype=np.uint64)
    table = zeta_sub_numpy(lift_coefficients(polys, Z, e, L))
    assert table.dtype == np.uint64
    mod = 1 << (e + 1)
    for b in range(2):
        for S in range(1 << L):
            w = [int(Z[b, l]) + (1 << e) * ((S >> l) & 1) for l in range(L)]
            assert int(table[b, 0, S]) % mod == polys[0].eval_int(w) % mod


def test_solver_refuses_levels_the_ring_cannot_certify():
    from padic import ddp

    with pytest.raises(ValueError, match="64-bit ring"):
        ddp.solve([Poly(1, {(1,): 1})], 1, max_e=65)


@requires_metal
def test_kernel_provenance_is_recorded():
    """A result must carry the kernel commit its numbers came from."""
    commit = _yates.kernel_commit()
    assert commit != "unknown" and len(commit) == 12
