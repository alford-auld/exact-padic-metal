# p-adic character networks on Metal, at p = 2

An implementation of Tomoki Mihara, *p-adic Character Neural Network*
([arXiv:2603.29905v1](https://arxiv.org/abs/2603.29905v1), math.NT; cs.LG,
31 Mar 2026) for `p = 2`, on Apple Silicon.

- **Algorithms 1–6** transcribed literally in pure Python, as the reference
  every fast path is checked against.
- **Algorithm 6** — the digit dynamic program for
  `e_max = max{ e : f has a common zero mod p^e }` — with its inner double loop
  collapsed onto the Metal Yates butterfly kernel.
- **§4** — the reduction from a character network to a polynomial system.
- **[`apps/zeta`](apps/zeta/README.md)** — Poincaré series of a typed-in
  polynomial system, with the rational generating function recovered.
- **[`apps/grokking`](apps/grokking/README.md)** — an exact two-unit solution
  to the Grokking task, and a **structural obstruction in §4's own reduction**.

```sh
uv sync
uv run --locked python -m pytest                     # 207 tests
uv run --locked python -m apps.zeta --preset cusp
uv run --locked python -m bench.run                  # the full validated run
```

Requires Apple Silicon and Metal for the `dense` backend; without it the
`numpy` backend runs the same algorithm on the CPU. **Reachable problem size is
`L ≤ 24` variables** (a `2^L`-entry table per polynomial, 512 MiB at `L = 24`)
and `max_e ≤ 64` levels.

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
`O(|I|·L²)` instead of `2^L`.

**At `e = 0` there is no shortcut and none should be expected** — it asks for
common roots in `{0,1}^L` of multilinear polynomials over `F_2`, which is
NP-hard. `2^L` *is* the algorithm there, and that is where the kernel earns its
place.

The two paths must produce identical survivor **sets** (not just counts — they
enumerate in different orders), and the test suite and the run both check it.

### Measurements

Every figure below is from
[`bench/results/m4-16gb-2026-09-28.json`](bench/results/m4-16gb-2026-09-28.json)
(commit `6990c9cd5341`, MacBook Air M4 / 16 GB, macOS 26.3, seed 20260928), and
`tests/test_docs_claims.py` fails if this table and that file disagree.

**Protocol.** Wall clock, `mx.eval()` forced inside every rep and
`mx.synchronize()` around the window, so MLX's laziness cannot collapse reps.
1 warmup rep; 20 timed reps below `L = 20`, 5 at or above. **No cooldown
between sizes and no thermal control** — these are warm back-to-back runs on a
fanless machine and should be read as such. Device copy is `yates.device_copy`
on the same array under the same protocol, **re-measured at every footprint**.

Level 0, `|I| = 4` polynomials:

| `L` | array | `ZETA_SUB` | device copy | passes | CPU transform | speedup |
|----:|------:|-----------:|------------:|-------:|--------------:|--------:|
| 18 | 8 MiB | 1.32 ms | 0.43 ms | 3.11× | 52.7 ms | 40× |
| 20 | 32 MiB | 3.53 ms | 1.39 ms | 2.54× | 238.5 ms | 68× |
| 24 | 512 MiB | 34.49 ms | 12.52 ms | 2.75× | — | — |

The `L`-stage transform costs 2.5–3.1 device copies at `L ≥ 18` — threadgroup
tiling fuses roughly ten of the stages into one memory pass.

**Copy bandwidth is not one number.** Measured on this machine it runs from
0.3 GB/s at a 32 KiB working set to 87 GB/s at 256 MiB, and only the largest
footprints approach the ceiling. Quoting a single denominator would be
meaningless; the per-footprint figures are in the results file, and at
`L ≥ 22` (128–512 MiB) it is 82–87 GB/s.

One level at `e = 2`, square system (`|I| = L`), `|Z| = 8`. The dense path ran
to completion at every `L` shown and produced the same survivor set as the
structural path:

| `L` | dense transform | structural `F_2` | ratio |
|----:|----------------:|-----------------:|------:|
| 8 | 0.6 ms | 0.13 ms | 5× |
| 14 | 8.9 ms | 0.17 ms | 53× |
| 22 | 1178.2 ms | 0.42 ms | 2800× |

---

## Poincaré series

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
reproduces every exact term. 14 classical presets, each verified against
exhaustive search to a depth that costs `2^(eL)` and so differs by preset; the
per-preset depth and exact-prefix length are tabulated in
[`apps/zeta/`](apps/zeta/README.md).

## Grokking, exactly — and an obstruction in §4

[`neelnanda-io/Grokking`](https://github.com/neelnanda-io/Grokking) trains a
transformer on `(a+b) mod 113` and it groks — memorises, then much later
generalises via a *character* algorithm. A p-adic character network has a
character as its activation, so the solution can just be written down:

```
y = ( χ(p^j(a+b)) − χ(0) ) / p^F ,    j = E−1,  F = j+m
```

Two hidden units, exact on every pair, for every prime tested including
`P = 113` (all 12769 pairs).

The same app carries the repository's main **negative result**. Because
`χ` lands in `1 + qZ_p`, every entry of every live `C'` column is `≡ 1 mod p`
while the `A` and `b` columns vanish, so the mod-`p` Jacobian of a §4 system
has

```
rank_{F_p} J  =  M          (the number of outputs)
```

**exactly** — independent of the width `D`, the input dimension `N`, and the
sample count. Measured across 11 instance shapes. The width you add is
precisely the width that cannot help: Algorithm 6 then branches by `p^(L−M)`
per level on exactly the systems the same paper's reduction generates. Two
repairs were measured and both fail. For character networks, construction beats
search, and that is structural rather than an engineering gap. Full derivation
and measurements in [`apps/grokking/`](apps/grokking/README.md).

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
bench/results/       committed measurement artifacts the READMEs cite
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

These have not been reported to the author. Anyone intending to build on the
paper should read `docs/deviations.md` before trusting the printed pseudocode.

## Licence and attribution

MIT — see [`LICENSE`](LICENSE). Attribution is in [`NOTICE`](NOTICE): the
paper implemented, the bundled kernel, dependency licences, and the
mathematical results the code depends on for its correctness. The two are
separate files because GitHub's licence detector reads `LICENSE` whole, and
anything appended to it makes the repository read as "NOASSERTION".

The Metal kernel in `vendor/yates/` is a verbatim copy of the `yates` package
from [`exact-yates-metal`](https://github.com/alford-auld/exact-yates-metal)
(MIT), pinned in `vendor/KERNEL_COMMIT`. It is vendored rather than a submodule
because run snapshots do not carry submodule contents. See
[`vendor/PROVENANCE.md`](vendor/PROVENANCE.md).

`exact-yates-metal` is the kernel; this repository is an application of it
beyond the chromatic-number app that ships there.
