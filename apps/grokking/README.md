# Grokking, exactly

[`neelnanda-io/Grokking`](https://github.com/neelnanda-io/Grokking) trains a
one-layer transformer on `(a + b) mod 113`. It memorises, then much later
generalises, and the mechanistic account of the generalising circuit is that it
computes with **characters** of `Z/113` — discrete Fourier components combined
by trigonometric identities.

A p-adic character network puts a character in the activation function. So for
this task the solution is not something to search for. It is two units wide and
you can write it down.

## The solution

```
y  =  ( chi(p^j (a + b))  −  chi(0) ) / p^F        j = E − 1,   F = j + m
```

with `chi = exp_p(q ·)`, `q = p^m`, `m = 2` at `p = 2` and `1` otherwise. In
the §4 variable layout that is `D = 2`:

```
A = [[p^j, p^j], [0, 0]]      b = [0, 0]      C = [p^-F, −p^-F]
```

**Why.** `chi(v) = 1 + q v + q²v²/2! + …`, and the term of index `e` has
valuation at least `(j+m)e − v_p(e!)`. So `chi(p^j u) = 1 + p^(j+m) u + O(p^(2(j+m) − 1/(p−1)))`,
and dividing by the linear coefficient leaves `u` with an error that vanishes
modulo `p^E` once `j ≥ E − 1`. Nothing is fitted: this is the homomorphism
property plus Legendre's formula.

**At the reference modulus.** `P = 113` is prime, so `p = 113`, `E = 1`,
`j = 0`, `F = 1`:

```
y = (chi(a + b) − 1) / 113            A = [[1,1],[0,0]],  C = [1/113, −1/113]
```

Verified against all `113² = 12769` pairs — zero mismatches. The same holds for
every prime tested (`p ∈ {2,3,5,7,113}`, `E ≤ 4`), and in each case the closed
form is checked to be a common zero of the polynomial system the §4 reduction
produces, not merely a correct predictor.

## And a negative result: Algorithm 6 cannot find it

The interesting part is what happens when you ask Algorithm 6 to *search* for
this solution instead of being handed it. It works at `E ≤ 2` and stalls at
`E = 3`, and the reason is structural rather than a tuning failure.

`chi' = q·chi` with `q = p^m` and `m ≥ 1`. So every derivative of the residual
with respect to an entry of `A` or `b` carries a factor of `p`, and those
columns of the mod-`p` Jacobian are **identically zero**. Measured at `p = 2`,
`E = 3`, with all 64 pairs as equations:

| level | \|Z\| | equations | L | F₂ rank | branching | columns live mod 2 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 128 | 64 | 8 | 1 | 2⁷ | `C'[0,0]`, `C'[0,1]` |
| 2 | 16384 | 64 | 8 | 1 | 2⁷ | `C'[0,0]`, `C'[0,1]` |
| 3 | 16384 | 64 | 8 | 1 | 2⁷ | `C'[0,0]`, `C'[0,1]` |

Sixty-four equations in eight unknowns, and the rank is **one**. The two live
columns are `C'[0,0]` and `C'[0,1]`, whose entries are `chi(u)` and `chi(0)` —
both `≡ 1 mod 2`, since `chi` lands in `1 + qZ_p`. Everything else is dead.

Consequences:

- **More data does not help.** The constraint gradient in `A` and `b` is zero,
  so extra pairs add rows of zeros.
- **`|Z|` multiplies by `p^(L−1)` every level**, here 128×, which overruns any
  survivor cap within three or four levels.
- **This is not specific to modular addition.** It follows from the shape of
  the §4 reduction, so it applies to every task in this framework. Modular
  addition only made it legible, because the closed form gives a solution known
  to be in the search space — so a miss is provably a search failure and not an
  absent solution.
- **It is not a kernel problem.** The subset-lattice transform runs only at
  level 0, where it takes 0.03 s of a solve that then spends minutes in the
  levels above. No change to the butterfly addresses any of this.

What would address it is an *implicit* representation of `Z`: the survivors at
each level form a coset of a rank-`(L−1)` subspace whose free directions are
always the same coordinates (`A` and `b`), so carrying `Z` as
`explicit C' digits × affine subspace` would collapse the branching instead of
enumerating it.

## Running it

```sh
uv run --locked python -m apps.grokking            # closed form + obstruction
uv run --locked python -m pytest tests/test_grokking.py
```

The `GROKKING` section of `bench/run.py` covers both halves.
