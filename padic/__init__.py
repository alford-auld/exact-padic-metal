"""Algorithm 6 of arXiv:2603.29905v1 on exact subset-lattice transforms.

    from padic.network import planted_instance
    from padic import ddp
    import random

    inst, _ = planted_instance(random.Random(0), n_samples=2, N=1, M=1, D=1, E=3)
    result = ddp.solve(inst.to_polynomial_system(), inst.n_vars, max_e=inst.E_star)
    print(result.summary())

The literal pseudocode lives in :mod:`padic.reference`; everything else is an
optimisation of it that the test suite holds to bitwise agreement.
"""

from . import character, ddp, gf2, network, poly, reference
from .ddp import Result, solve
from .network import NetworkInstance
from .poly import Poly

__all__ = [
    "character",
    "ddp",
    "gf2",
    "network",
    "poly",
    "reference",
    "NetworkInstance",
    "Poly",
    "Result",
    "solve",
]
