"""The Grokking task, as a p-adic character-network instance.

`neelnanda-io/Grokking <https://github.com/neelnanda-io/Grokking>`_ trains a
one-layer transformer on ``(a + b) mod 113`` from a fraction of the ``113^2``
pairs, and the model generalises long after it has fit the training split.

What is reproduced here is the **task and its split**, not the architecture:
the model is the §4 character network ``C chi(A x + b)``, and the fitting
procedure is Algorithm 6 rather than gradient descent.  The two are compared on
what they produce, not on how they get there.

The task modulus must be a prime power ``p^E`` for the character to be a
character *of the right group*.  ``P = 113`` is prime, so it is ``p = 113,
E = 1``; the 2-adic case is ``p = 2``, modulus ``2^E``, which is where the
subset-lattice kernel applies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from padic.network import NetworkInstance

Pair = Tuple[int, int]


@dataclass(frozen=True)
class ModularAddition:
    """``(a, b) -> (a + b) mod p^E``, over all ``p^E x p^E`` pairs."""

    p: int
    E: int
    #: Hidden width.  Two is what the closed form needs; larger only adds
    #: unknowns, and the search cost is exponential in them.
    D: int = 2

    @property
    def modulus(self) -> int:
        return self.p**self.E

    @property
    def n_pairs(self) -> int:
        return self.modulus**2

    def all_pairs(self) -> List[Pair]:
        P = self.modulus
        return [(a, b) for a in range(P) for b in range(P)]

    def target(self, a: int, b: int) -> int:
        return (a + b) % self.modulus

    def split(self, rng: random.Random, train_fraction: float) -> Tuple[List[Pair], List[Pair]]:
        """A uniform random train/test split, as in the reference repository."""
        pairs = self.all_pairs()
        rng.shuffle(pairs)
        cut = max(1, round(train_fraction * len(pairs)))
        return pairs[:cut], pairs[cut:]

    def instance(self, pairs: Sequence[Pair], F: int) -> NetworkInstance:
        """The §4 instance whose feasibility is the fitting problem.

        ``F`` is the bound on the valuation of the denominators of ``C``; the
        closed form needs ``F = E - 1 + m``, and the search must be given at
        least that much or the solution is outside the search space.
        """
        X = tuple((a, b) for a, b in pairs)
        Y = tuple(((self.p**F * self.target(a, b)),) for a, b in pairs)
        return NetworkInstance(X, Y, D=self.D, E=self.E, F=F, p=self.p)


#: The task as the reference repository poses it.
GROKKING = ModularAddition(p=113, E=1)
