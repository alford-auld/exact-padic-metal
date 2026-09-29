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

**Why.** `chi(v) = 1 + qv + q²v²/2! + …`. Feeding it `v = p^j u`, the term of
index `e` has valuation

```
v_p( q^e (p^j u)^e / e! )  ≥  (j+m)e − (e − s_p(e))/(p−1)      [Legendre]
```

The binding term is `e = 2`, where this is `2(j+m) − v_p(2!)` — and `v_p(2!)`
is the place the two primes part company: it is `1` at `p = 2` and `0` for odd
`p`. Dividing through by the linear coefficient `p^(j+m)` leaves `u` plus an
error of valuation at least

```
(j+m) − v_p(2!)  =  j + m − 1   (p = 2, m = 2)
                 =  j + m       (odd p, m = 1)
```

and with `j = E − 1` both are exactly `E`, so the error vanishes mod `p^E`.
Terms with `e ≥ 3` are not binding: the same bound gives
`(e−1)·[(j+m) − 1/(p−1)]`, which is increasing in `e` and already exceeds `E`
at `e = 3` for both cases.

Using the uniform bound `O(p^(2(j+m) − 1/(p−1)))` instead would be tight at
`p = 2`, where `1/(p−1) = 1 = v_2(2!)`, and *loose for odd `p`*, where it
throws away `v_p(2!) = 0` and yields only `E − 1/(p−1) < E`. The conclusion
holds at every prime; the uniform bound only demonstrates it at the one prime
that does not need the slack.

Nothing is fitted: this is the homomorphism property plus Legendre's formula.

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
this solution instead of being handed it. It returns a verified fit at
`E ≤ 2` and stalls at `E = 3`, and the reason is structural rather than a
tuning failure.

`E ≤ 2` is not the search working. Branching is `2^7` per level at every
level, so `|Z|` runs `1 → 128 → 16384 → cap`; `E = 1` and `E = 2` need three
and five levels respectively and a survivor of the right coset happens to
remain inside the cap, while `E = 3` needs seven and the cap discards the true
solution's ancestor at level 5 (`→ 0`). Those runs are `lower_bound`, not
`exact`; what makes their answers trustworthy is that the recovered parameter
vector is checked directly against every pair, not that the search terminated
cleanly.

`chi' = q·chi` with `q = p^m` and `m ≥ 1`, so every derivative of the residual
with respect to an entry of `A` or `b` carries a factor of `p`: those columns
of the mod-`p` Jacobian are **identically zero**.

What is left is sharper than "few columns survive". Since `chi` lands in
`1 + qZ_p`, *every entry* of *every* live `C'` column is `≡ 1 mod p`. Within
one output row the `C'` block is therefore all-ones and contributes rank
exactly one, and distinct outputs occupy disjoint columns, so

```
rank_{F_p} J  =  M          (the number of outputs)
```

**exactly** — independent of the hidden width `D`, the input dimension `N`,
and the sample count `|I|`. Measured across 11 instance shapes
(`N ≤ 4`, `M ≤ 4`, `D ≤ 4`, `|I| ≤ 128`): rank equals `M` in every one, with
only `C'` columns ever live. The width you add is exactly the width that
cannot help.

Measured at `p = 2`, `E = 3`, `M = 1`, with all 64 pairs as equations:

| level | \|Z\| | equations | L | F₂ rank | branching | columns live mod 2 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 128 | 64 | 8 | 1 | 2⁷ | `C'[0,0]`, `C'[0,1]` |
| 2 | 16384 | 64 | 8 | 1 | 2⁷ | `C'[0,0]`, `C'[0,1]` |
| 3 | 16384 | 64 | 8 | 1 | 2⁷ | `C'[0,0]`, `C'[0,1]` |

Sixty-four equations in eight unknowns, and the rank is **one** — which is
`M`, as above. The two live columns are `C'[0,0]` and `C'[0,1]`, whose entries
are `chi(u)` and `chi(0)`, both `≡ 1 mod 2`. Everything else is dead.

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

### Two repairs, both measured, both failing

**Reschedule the digits.** The `A` and `b` digits are not unconstrained, only
constrained `m` levels later — `∂f/∂A` is divisible by `p^m`, so bumping digit
`e` moves the residual at valuation `e+m`. Staggering the schedule so each
variable is decided when it first bites does revive the dead columns: measured
rank goes `1 → 3` at the Grokking shape. But branching only falls `128 → 32`,
a factor of 4 against a gap of ~10³, and the ceiling is structural — quadrupling
the data leaves the staggered rank at 3, and widening the network raises the
nullity without raising the rank.

**Represent `Z` instead of enumerating it.** The survivors do form a single
coset — for the first two levels. Measured coset counts: `1, 1, ≥8, ≥256`,
growing by at least 32× per level thereafter, at which point a coset
representation has converged on enumeration and buys nothing. (These are sound
lower bounds: the stabiliser estimate used is an over-estimate, so the true
counts can only be larger.)

The two failures have one cause. The survivor set is affine *exactly while the
system is degenerate* — at low precision `chi ≡ 1 mod p`, the rank is `M`, and
nothing is constrained, which is why everything looks linear. As soon as
`chi`'s nonlinear terms start to bite and could constrain the parameters, the
affine structure goes with them. Structure without constraint, or constraint
without structure; never both.

## Running it

```sh
uv run --locked python -m apps.grokking            # closed form + obstruction
uv run --locked python -m pytest tests/test_grokking.py
```

The `GROKKING` section of `bench/run.py` covers both halves.
