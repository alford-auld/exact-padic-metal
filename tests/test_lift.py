"""The multilinear lift is the bridge between Algorithm 6 and the transform.

If these fail, nothing downstream means anything.
"""

import numpy as np
import pytest

from conftest import random_system
from padic.ddp import zeta_sub_numpy
from padic.poly import lift_coefficients


@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("e", [0, 1, 2, 3, 5])
def test_zeta_of_lift_is_the_value_table(rng, L, e):
    """(zeta_sub c)(S) == f(z + 2^e 1_S), exactly, for every z and every S."""
    polys = random_system(rng, L, 3)
    B = 5
    Z = np.array(
        [[rng.randrange(1 << e) if e else 0 for _ in range(L)] for _ in range(B)],
        dtype=np.uint64,
    )
    coeffs = lift_coefficients(polys, Z, e, L)
    table = zeta_sub_numpy(coeffs)
    mod = 1 << (e + 1)

    for b in range(B):
        for S in range(1 << L):
            w = [int(Z[b, l]) + (1 << e) * ((S >> l) & 1) for l in range(L)]
            for i, f in enumerate(polys):
                assert int(table[b, i, S]) % mod == f.eval_int(w) % mod


@pytest.mark.parametrize("e", [1, 2, 3, 4])
def test_lift_is_affine_in_the_digits_above_level_zero(rng, e):
    """For e >= 1 every coefficient on two or more digits vanishes mod 2^(e+1).

    This is what licenses the structural path: terms with two digit factors
    carry 2^(2e), and 2e >= e+1.
    """
    L = 4
    polys = random_system(rng, L, 3, max_deg=4)
    Z = np.array(
        [[rng.randrange(1 << e) for _ in range(L)] for _ in range(6)], dtype=np.uint64
    )
    coeffs = lift_coefficients(polys, Z, e, L)
    mod = np.uint64((1 << (e + 1)) - 1)
    for S in range(1 << L):
        if bin(S).count("1") >= 2:
            assert np.all((coeffs[:, :, S] & mod) == 0), f"S={S} is nonzero at e={e}"


def test_lift_at_level_zero_is_genuinely_multilinear(rng):
    """At e = 0 the high-order coefficients do *not* vanish.

    The counterpart to the test above: the affine shortcut is unavailable at
    e = 0, which is why the transform is needed there.
    """
    L = 4
    polys = [random_system(rng, L, 1, max_deg=4, nterms=8)[0] for _ in range(4)]
    Z = np.zeros((1, L), dtype=np.uint64)
    coeffs = lift_coefficients(polys, Z, 0, L)
    high = [S for S in range(1 << L) if bin(S).count("1") >= 2]
    assert np.any(coeffs[:, :, high] & np.uint64(1))


def test_singleton_coefficient_equals_the_unit_step(rng):
    """coeff[{l}] == f(z + 2^e e_l) - f(z); the structural path relies on it."""
    L, e = 4, 2
    polys = random_system(rng, L, 3)
    Z = np.array([[rng.randrange(1 << e) for _ in range(L)] for _ in range(5)],
                 dtype=np.uint64)
    coeffs = lift_coefficients(polys, Z, e, L)
    for b in range(Z.shape[0]):
        z = [int(v) for v in Z[b]]
        for i, f in enumerate(polys):
            base = f.eval_int(z)
            for l in range(L):
                w = list(z)
                w[l] += 1 << e
                expected = (f.eval_int(w) - base) % (1 << 64)
                assert int(coeffs[b, i, 1 << l]) == expected
