# p-adic character networks on Metal, at p = 2

An implementation of T. Mihara, *p-adic Character Neural Network*
([arXiv:2603.29905v1](https://arxiv.org/abs/2603.29905), math.NT, 31 Mar 2026)
for `p = 2`, on Apple Silicon.

- **Algorithms 1–6** transcribed literally in pure Python, as the reference
  every fast path is checked against.
- **Algorithm 6** — the digit dynamic program for
  `e_max = max{ e : f has a common zero mod p^e }` — with its inner double loop
  collapsed onto the Metal Yates butterfly kernel.
- **§4** — the reduction from a character network to a polynomial system.
- Two things to play with: a **Poincaré series** explorer, and an exact
  solution to the **Grokking** task.

```sh
uv sync
uv run --locked python -m pytest                     # 183 tests
uv run --locked python -m apps.zeta --preset cusp    # play
uv run --locked python -m bench.run                  # the full validated run
```

Requires Apple Silicon and Metal for the `dense` backend; without it the
`numpy` backend runs the same algorithm on the CPU.

---

## Why the kernel fits

Algorithm 6 keeps the set `Z` of common zeros mod `p^e`, lifts each `z` by
every digit vector `d ∈ N_{<p}^L` to `w = z + p^e d`, and keeps what vanishes
mod `p^(e+1)`. At `p = 2` the digit vector `d ∈ {0,1}^L` **is a subset** of the
`L` coordinates, so one level lives on the Boolean lattice `2^[L]`.

**The lift is multilinear.** Since `d_l ∈ {0,1}`, `d_l^k = d_l` *as integers*,
so a monomial collapses by interpolation:

```
(z_l + P·d_l)^k  =  z_l^k  +  d_l·((z_l + P)^k − z_l^k)
```

and the table of all `2^L` values is the **subset zeta transform** of the
coefficient vector:

```
(ZETA_SUB c)(S) = Σ_{T ⊆ S} c_T = f(z + 2^e·1_S)
```

Lines 7–12 of the pseudocode — the whole inner double loop — become one batched
`ZETA_SUB`.

**Wraparound costs nothing.** The kernel is exact over `Z/2^64`, and
`Z/2^64 ↠ Z/2^(e+1)` is a ring surjection for every `e+1 ≤ 64`, so reducing its
output *is* the answer. `ZETA_SUB` is unipotent, so no bits are lost. See
[`docs/exactness.md`](docs/exactness.md).

## Two paths, cross-checked

| backend | level 0 | levels ≥ 1 |
|---|---|---|
| `structural` *(default)* | Metal `ZETA_SUB` | `F_2` elimination |
| `dense` | Metal `ZETA_SUB` | Metal `ZETA_SUB` |
| `numpy` | CPU transform | CPU transform |

For `e ≥ 1` the lift is **affine** in `d` (terms with two digit factors carry
`2^(2e)`, and `2e ≥ e+1`), so the survivors solve an affine `F_2` system —
`O(|I|·L²)` instead of `2^L`. Measured at `e = 2`: 3× faster at `L = 8`, 28× at
`L = 14`, ~2700× at `L = 22`.

**At `e = 0` there is no shortcut and none should be expected** — it asks for
common roots in `{0,1}^L` of multilinear polynomials over `F_2`, which is
NP-hard. `2^L` *is* the algorithm there, and that is where the kernel earns its
place. Measured at level 0 on an M4:

| `L` | array | `ZETA_SUB` | device copy | CPU transform |
|----:|------:|-----------:|------------:|--------------:|
| 18 | 8 MiB | 1.13 ms | 0.49 ms | 54.3 ms |
| 20 | 32 MiB | 2.29 ms | 1.23 ms | 241.3 ms |
| 24 | 512 MiB | 35.3 ms | 12.1 ms | — |

The whole `L`-stage transform costs about three copies of the array against a
measured 89 GB/s device copy — threadgroup tiling fuses up to 11 of the 24
stages into one memory pass.

The two paths must produce identical survivor **sets** (not just counts — they
enumerate in different orders), and the test suite and the run both check it.

## What a result means

`solve()` returns a `guarantee`, and it is the field to read first.

| `guarantee` | meaning |
|---|---|
| `exact` | the program ran to a genuinely empty `W`; `e_max` is the true maximum |
| `lower_bound` | it was stopped by the level bound or the survivor cap; a zero mod `2^e_max` exists, nothing is claimed above |

Both stopping conditions can only make the answer *too small*. `max_e > 64` is
refused rather than silently approximated. Details in
[`docs/exactness.md`](docs/exactness.md).

---

## Play: Poincaré series

```console
$ uv run --locked python -m apps.zeta "x*y" --verify
Poincare series   sum_e N_e t^e  =  (1 - t) / (1 - 4*t + 4*t^2)
recurrence        N_e = +4 N_(e-1)  -4 N_(e-2)   (order 2)
verify  exhaustive search over (Z/2^e)^2 for e <= 7: AGREE
verify  closed form node: N_e = (e+2)2^(e-1): AGREE
```

The survivor counts of Algorithm 6 **are** `N_e = #{x mod 2^e : f(x) = 0}`, the
coefficients of the Poincaré series. Igusa proved the series is rational, so a
few terms give all of it — the playground recovers it and checks the fit
reproduces every term. 14 classical presets (`--list`), all verified against
exhaustive search. See [`apps/zeta/`](apps/zeta/README.md).

## Play: Grokking, exactly

[`neelnanda-io/Grokking`](https://github.com/neelnanda-io/Grokking) trains a
transformer on `(a+b) mod 113` and it groks — memorises, then much later
generalises via a *character* algorithm. A p-adic character network has a
character as its activation, so the solution can just be written down:

```
y = ( χ(p^j(a+b)) − χ(0) ) / p^F ,    j = E−1,  F = j+m
```

Two hidden units, exact on every pair, for every prime tested including
`P = 113` (all 12769 pairs). See [`apps/grokking/`](apps/grokking/README.md).

That app also carries a **negative result** about the paper: §4's reduction
produces systems whose mod-`p` Jacobian has rank ≤ ~`M(N+1)` *independent of
the sample count*, with nullity growing in the network width — because
`χ' = qχ` with `q = p^m` makes the `A` and `b` columns vanish mod `p`. So
Algorithm 6 branches by `p^(L−r)` per level on exactly the systems the same
paper's reduction generates. Two repairs were measured and both fail
(rescheduling: 4× against a ~10³× gap; coset representation: exact for two
levels, ≥256 cosets by level four). For character networks, construction beats
search, and that is structural rather than an engineering gap.

---

## Layout

```
padic/reference.py   literal pseudocode, Algorithms 1-6, pure Python
padic/poly.py        sparse polynomials over Z; the multilinear lift
padic/ddp.py         Algorithm 6: dense (transform) and structural (F_2) paths
padic/gf2.py         affine systems over F_2
padic/character.py   chi as a polynomial (Taylor coefficients)
padic/network.py     the §4 reduction, and instance generators
apps/zeta/           Poincare series playground
apps/grokking/       exact modular addition, and the obstruction
bench/run.py         the run command: validate, then measure
docs/exactness.md    what a reported e_max means
docs/deviations.md   three faults in the printed pseudocode, and the fixes
vendor/yates/        the Metal kernel, vendored (see vendor/PROVENANCE.md)
```

## Deviations from the paper

Three, documented and pinned by tests in
[`docs/deviations.md`](docs/deviations.md):

1. **Algorithm 5 as printed cannot compute `a^x`** — it initialises `y ← 0`,
   never reads `a`, and squares the exponent rather than a running base.
2. **Algorithm 4's precision bound drops digits the answer depends on.** The
   obvious fix — stopping once the bound hits zero — is *also* wrong, because
   `v + me = e + s₂(e)` is not monotonic in `e`.
3. **§4's `|p|^(e_max − 1)`** needs a reading to be consistent with Algorithm
   6's own caption.

## Licence and attribution

MIT — see [`LICENSE`](LICENSE), which also records the full attribution: the
paper implemented, the bundled kernel, dependency licences, and the
mathematical results the code depends on for its correctness.

The Metal kernel in `vendor/yates/` is a verbatim copy of the `yates` package
from [`exact-yates-metal`](https://github.com/alford-auld/exact-yates-metal)
(MIT), pinned in `vendor/KERNEL_COMMIT`. It is vendored rather than a
submodule because run snapshots do not carry submodule contents. See
[`vendor/PROVENANCE.md`](vendor/PROVENANCE.md).

`exact-yates-metal` is the kernel; this repository is an application of it
beyond the chromatic-number app that ships there.
