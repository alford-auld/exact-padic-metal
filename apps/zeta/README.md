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

## Some things it finds

| preset | system | Poincaré series | dim |
|---|---|---|---:|
| `square` | `x²` | `(1 + t) / (1 − 2t²)` | 0.5 |
| `cube` | `x³` | `(1 + t + 2t²) / (1 − 4t³)` | 0.67 |
| `node` | `xy` | `(1 − t) / (1 − 4t + 4t²)` | 1 |
| `sum2` | `x² + y²` | `1 / (1 − 2t)` | 1 |
| `d4` | `x³ − xy²` | `(1 + t + 4t²) / (1 − 2t − 16t³ + 32t⁴)` | 1.35 |
| `tacnode` | `y² − x⁴` | `(1 + 4t² + 8t³ − 16t⁴) / (1 − 2t − 32t⁴ + 64t⁵)` | 1.25 |
| `whitney` | `x² − y²z` | `(1 − 8t²) / (1 − 4t − 16t² + 64t³)` | 2 |
| `sum3` | `x² + y² + z²` | `(1 + 4t) / (1 − 8t²)` | 1.5 |
| `cusp` | `y² − x³` | `(1 + 2t² − 64t⁶) / (1 − 2t − 128t⁶ + 256t⁷)` | 1.18 |

The cusp needs `--upto 18`: its recurrence has order 7, and the fit demands
enough terms to be over-determined rather than merely consistent.

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
