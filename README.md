# Algorithm 6 on exact subset-lattice transforms

Algorithm 6 of T. Mihara, *p-adic Character Neural Network*
([arXiv:2603.29905v1](https://arxiv.org/abs/2603.29905), math.NT, 31 Mar 2026),
implemented at `p = 2` on the Metal Yates butterfly kernel from
[`exact-yates-metal`](vendor/exact-yates-metal) — plus the §4 reduction that
produces its input from an actual 2-adic character network.

## Why a subset-lattice transform belongs here

Algorithm 6 answers: what is the largest `e` such that a polynomial system
`f = (f_i)_{i in I}` over `Z` has a common zero in `Z/p^e Z`? It keeps the set
`Z` of zeros modulo `p^e`, lifts each by every digit vector
`d in N_{<p}^L` to `w = z + p^e d`, and keeps what vanishes modulo `p^(e+1)`.

At `p = 2` the digit vector `d in {0,1}^L` **is a subset** of the `L`
coordinates. One level of the dynamic program is therefore a computation over
the Boolean lattice `2^[L]` — the kernel's native domain. Two facts make the
connection exact rather than decorative:

**1. The lift is multilinear.** Because `d_l in {0,1}`, we have `d_l^k = d_l`
*as integers*, so `f(z + 2^e d)` collapses to a multilinear polynomial in `d`:

```
(z_l + P d_l)^k  ==  z_l^k  +  d_l ((z_l + P)^k - z_l^k)
```

and the table of all `2^L` values is the **subset zeta transform** of its
coefficient vector:

```
(ZETA_SUB c)(S) = sum_{T subset S} c_T = f(z + 2^e 1_S)
```

So lines 7–12 of the pseudocode — the whole inner double loop — become one
batched `ZETA_SUB`.

**2. Wraparound costs nothing.** The kernel is exact over `Z/2^64`, and
`Z/2^64 ->> Z/2^(e+1)` is a ring surjection for every `e+1 <= 64`, so reducing
its output *is* the answer. `ZETA_SUB` is unipotent, so it loses no bits
either. See [`docs/exactness.md`](docs/exactness.md).

## Where the transform is load-bearing, and where it is not

This is the honest version of the story, and the benchmark measures both
halves.

For `e >= 1` the lift is **affine** in `d`: terms with two or more digit
factors carry `2^(2e)`, and `2e >= e+1`. Writing `f_i(z) = 2^e c_i`, the
surviving digits are the solutions of an affine system over `F_2`,

```
sum_l g_{i,l} d_l = c_i,     g_{i,l} = (coefficient of d_l) / 2^e
```

which Gaussian elimination settles in `O(|I| L^2)` instead of `2^L`. Measured
on an M4, at level `e = 2` with a square system:

| `L` | dense transform | structural `F_2` | ratio |
|----:|----------------:|-----------------:|------:|
|   8 |          0.4 ms |          0.13 ms |  3.3x |
|  14 |          4.0 ms |          0.17 ms | 23x   |
|  19 |          387 ms |          0.58 ms | 665x  |
|  22 |         1173 ms |          0.37 ms | 3141x |

**At `e = 0` there is no such shortcut, and none should be expected**: the step
asks for the common roots in `{0,1}^L` of a system of multilinear polynomials
over `F_2`, which is NP-hard. `2^L` *is* the algorithm there, and that is
exactly where the kernel earns its place. Measured at level 0, `|I| = 4`:

| `L` | array | `ZETA_SUB` | device copy | passes | CPU transform | speedup |
|----:|------:|-----------:|------------:|-------:|--------------:|--------:|
|  18 |  8 MiB |    1.13 ms |     0.42 ms |  2.7x |       53.1 ms |     47x |
|  20 | 32 MiB |    2.58 ms |     1.05 ms |  2.4x |      237.6 ms |     92x |
|  24 | 512 MiB |   33.8 ms |     11.7 ms |  2.9x |             — |       — |

The whole `L`-stage transform costs about **three copies of the array** —
threadgroup tiling fuses up to 8 of the 24 stages into a single memory pass —
against a measured device-to-device copy bandwidth of 92 GB/s on this machine.

So the default backend, `"structural"`, uses the transform at level 0 and `F_2`
elimination above it. `"dense"` uses the transform at every level and exists to
cross-check the other; the two must produce identical survivor *sets*, and the
test suite and the run both check that they do.

## The real limit: `|Z|`, not the cost of a level

Each surviving `z` lifts to up to `2^(L - rank)` successors, so an
under-determined system multiplies `|Z|` at every level. Measured growth:

```
 L  |I|   |Z_e| by level
 8    2   64 -> 4096 -> 262144 -> 4194304 (capped)
 8    8   24 -> 128 -> 384 -> 768 -> 2048 -> 2048
12   12   32 -> 64 -> 128 -> 256 -> 512 -> 1024
```

This is inherent to Algorithm 6 as written, not to this implementation. A
`max_survivors` cap keeps it bounded, and a capped run reports
`guarantee = "lower_bound"` — a one-sided claim, since discarding survivors can
only make `e_max` too small.

## Phase 2: from a network to the system

§4 reduces the character network

```
minimise || (y_i - C chi(A x_i + b))_{i in I} ||    over A, b integral, C in Q_p
```

to a polynomial system. Multiplying by `p^F` clears the only denominators
(`C' = p^F C`), and `chi = exp_p(q .)` truncates to a polynomial modulo
`p^(E+F)`, so with `L = D N + D + M D` unknowns,

```
f_{i,j} = p^F y_{i,j} - sum_d C'_{j,d} chi( sum_k A_{d,k} x_{i,k} + b_d )
```

is a polynomial over `Z`. That `A` and `b` stay integral — the point of using a
character rather than a ball indicator — is what makes this a polynomial system
at all.

```python
import random
from padic import ddp
from padic.network import planted_instance

inst, params = planted_instance(random.Random(0), n_samples=2, N=1, M=1, D=1, E=3)
polys = inst.to_polynomial_system()
res = ddp.solve(polys, inst.n_vars, max_e=inst.E_star)
print(res.summary())                          # e_max >= 3  (level bound ...)
print(inst.objective_valuation(res.e_max))    # minimum objective = |2|^3
```

`NetworkInstance.residual_valuation` re-evaluates the residual through `chi`
itself rather than through the polynomial approximation, so it can catch an
error in the reduction instead of repeating it.

## Layout

```
padic/reference.py   literal pseudocode, Algorithms 1-6, pure Python
padic/poly.py        sparse polynomials over Z; the multilinear lift
padic/ddp.py         Algorithm 6: dense (transform) and structural (F_2) paths
padic/gf2.py         affine systems over F_2
padic/character.py   chi as a polynomial (Taylor coefficients)
padic/network.py     the §4 reduction, and instance generators
bench/run.py         the run command: validate, then measure
docs/exactness.md    what a reported e_max means
docs/deviations.md   three faults in the printed pseudocode, and the fixes
```

Three deviations from the paper are documented and tested — Algorithm 5 as
printed cannot compute `a^x`, Algorithm 4's precision bound drops digits the
answer depends on, and §4's `|p|^(e_max - 1)` needs a reading. See
[`docs/deviations.md`](docs/deviations.md).

## Running it

```sh
git submodule update --init          # the Metal kernel
uv sync
uv run --locked python -m pytest     # 116 tests
uv run --locked python -m bench.run  # the full run, ~80 s on an M4
```

Requires Apple Silicon and Metal for the `"dense"` backend. Without it the
`"numpy"` backend runs the same algorithm on the CPU and the tests skip the
GPU-specific checks.
