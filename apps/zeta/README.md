# Poincaré series playground

Type a polynomial at it and get the series.

```console
$ uv run --locked python -m apps.zeta "x*y" --verify
system      x*y = 0
variables   x, y   (L = 2)
degree      2      terms 1

N_e = #{x mod 2^e : f(x) = 0}   [structural, 0.00s]
  N_0  =              1
  N_1  =              3   x3.000
  N_2  =              8   x2.667
  ...
  N_8  =          1,280   x2.222

growth  N_(e+1)/N_e -> 2^1.00   (dimension of the zero locus over Z_2)

Poincare series   sum_e N_e t^e  =  (1 - t) / (1 - 4*t + 4*t^2)
recurrence        N_e = +4 N_(e-1)  -4 N_(e-2)   (order 2)

verify  exhaustive search over (Z/2^e)^2 for e <= 7: AGREE
verify  closed form node (two crossing lines): N_e = (e+2)2^(e-1): AGREE
```

`uv run --locked python -m apps.zeta --list` shows the presets.

## What it computes

For a system `f` over `Z`, the **Poincaré series** at `p = 2` is

```
P(t) = Σ_e N_e t^e ,     N_e = #{ x ∈ (Z/2^e Z)^L : f(x) = 0 }
```

The survivor counts of Algorithm 6 **are** the `N_e`: every common zero modulo
`2^(e+1)` reduces to one modulo `2^e`, so the digit dynamic program enumerates
exactly the zero set at each level. Level 0 is the `2^L` subset-lattice
transform on the Metal kernel; levels above it are `F_2` elimination.

Igusa proved `P(t)` is **rational**, so a handful of terms determines the whole
series. The playground recovers it by finding the shortest constant-coefficient
recurrence the exact terms admit, over `Fraction`, and then verifying the fit
reproduces every term it was given.

## Every preset, and how deeply each was checked

From [`bench/results/m4-16gb-2026-09-28.json`](../../bench/results/m4-16gb-2026-09-28.json);
`tests/test_docs_claims.py` fails if this table and that file disagree.

| preset | L | terms | exact | brute≤ | order | system | Poincaré series | dim |
|---|--:|--:|--:|--:|--:|---|---|--:|
| `cube` | 1 | 12 | 12 | 8 | 3 | `x^3` | `(1 + t + 2*t^2) / (1 - 4*t^3)` | 0.67 |
| `square` | 1 | 12 | 12 | 8 | 2 | `x^2` | `(1 + t) / (1 - 2*t^2)` | 0.50 |
| `cusp` | 2 | 18 | 18 | 6 | 7 | `y^2 - x^3` | `(1 + 2*t^2 - 64*t^6) / (1 - 2*t - 128*t^6 + 256*t^7)` | 1.17 |
| `d4` | 2 | 18 | 15 | 6 | 4 | `x^3 - x*y^2` | `(1 + t + 4*t^2) / (1 - 2*t - 16*t^3 + 32*t^4)` | 1.33 |
| `ec` | 2 | 18 | 18 | 6 | 3 | `y^2 - x^3 - x - 1` | `(1 - 2*t^2) / (1 - 2*t)` | 1.00 |
| `line-pair` | 2 | 18 | 18 | 6 | 2 | `x*y , x + y` | `(1 + t) / (1 - 2*t^2)` | 0.50 |
| `node` | 2 | 18 | 18 | 6 | 2 | `x*y` | `(1 - t) / (1 - 4*t + 4*t^2)` | 1.00 |
| `pell` | 2 | 18 | 18 | 6 | 1 | `x^2 - 2*y^2` | `(1) / (1 - 2*t)` | 1.00 |
| `sing-ec` | 2 | 18 | 18 | 6 | — | `y^2 - x^3 - x^2` | *needs more terms* | 1.16 |
| `sum2` | 2 | 18 | 18 | 6 | 1 | `x^2 + y^2` | `(1) / (1 - 2*t)` | 1.00 |
| `tacnode` | 2 | 18 | 16 | 6 | 5 | `y^2 - x^4` | `(1 + 4*t^2 + 8*t^3 - 16*t^4) / (1 - 2*t - 32*t^4 + 64*t^5)` | 1.25 |
| `fermat3` | 3 | 12 | 10 | 4 | 4 | `x^3 + y^3 + z^3` | `(1 + 4*t^2 - 32*t^3) / (1 - 4*t - 64*t^3 + 256*t^4)` | 2.00 |
| `sum3` | 3 | 12 | 12 | 4 | 2 | `x^2 + y^2 + z^2` | `(1 + 4*t) / (1 - 8*t^2)` | 1.50 |
| `whitney` | 3 | 12 | 10 | 4 | 3 | `x^2 - y^2*z` | `(1 - 8*t^2) / (1 - 4*t - 16*t^2 + 64*t^3)` | 2.00 |

The three columns that qualify the rest:

- **terms** — how many `N_e` were computed.
- **exact** — how many of them are below the survivor cap and so are *counts*
  rather than lower bounds. **The fit uses only these.** `d4`, `tacnode`,
  `fermat3` and `whitney` exceed the cap before the last term, so their series
  come from a shorter prefix than the term count suggests.
- **brute≤** — the depth to which exhaustive search over all of `(Z/2^e)^L`
  confirmed the counts. It costs `2^(eL)`, so it necessarily differs by `L`:
  `e ≤ 8` at one variable, `e ≤ 6` at two, `e ≤ 4` at three. "Verified against
  exhaustive search" means *to this depth*, not to the end of the table.

`cusp` needs all 18 exact terms because its recurrence has order 7, and the fit
demands enough equations to be over-determined rather than merely consistent.
`sing-ec` is the one preset that does not resolve at 18 terms; Igusa's theorem
says a rational form exists, so this is a statement about the budget, not about
the series.

`dim` is `log₂` of the dominant root of the recurrence — the growth rate of
`N_e`, i.e. the dimension of the zero locus over `Z_2`. It is read off the
recurrence rather than from a sliding ratio, because these series are often
periodic: `x²` alternates ratios 1, 2, 1, 2, and a short window would report
the phase instead of the growth.

## Two things worth noticing

**Every preset is checked against exhaustive search** over all of `(Z/2^e)^L`
(`--verify`), and the three with classical closed forms are checked against
those too. The dynamic program never gets the benefit of the doubt.

**The cap is visible, and the series sees past it.** Survivor sets are capped
at `--max-survivors`; beyond that point `N_e` is a lower bound rather than a
count, and the output says so. But the series fitted from the *exact* prefix
keeps going, which is Igusa's rationality doing real work:

```
  N_10 =      3,670,016   x4.667
  N_11 =      4,194,304  -> 14,680,064 from the fitted series
  N_12 =      4,194,304  -> 67,108,864 from the fitted series
```

## Limits

- `p = 2` only, like the rest of the repository.
- `L ≤ 24` for the transform path (a `2^L` table per polynomial); the `F_2`
  path above level 0 goes further, but level 0 has to run first.
- This is brute lifting, not resolution of singularities. For the small,
  well-studied examples above, symbolic methods (Denef's formula) are the
  right tool and will beat it. What this does is run on systems with more
  variables than a resolution is practical for, and check itself against
  brute force wherever brute force can still be computed.
