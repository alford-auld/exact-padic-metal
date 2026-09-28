import random

import pytest

from padic.poly import Poly


def random_poly(rng: random.Random, nvars: int, max_deg: int = 3,
                nterms: int = 4, cmax: int = 24) -> Poly:
    terms = {}
    for _ in range(nterms):
        exps = [0] * nvars
        for _ in range(rng.randint(0, max_deg)):
            exps[rng.randrange(nvars)] += 1
        c = rng.randint(-cmax, cmax)
        if c:
            key = tuple(exps)
            terms[key] = terms.get(key, 0) + c
    return Poly(nvars, terms)


def random_system(rng: random.Random, nvars: int, npolys: int, **kw):
    return [random_poly(rng, nvars, **kw) for _ in range(npolys)]


@pytest.fixture
def rng():
    return random.Random(20260928)


def has_metal() -> bool:
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


requires_metal = pytest.mark.skipif(not has_metal(), reason="no Metal device")
