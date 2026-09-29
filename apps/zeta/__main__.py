"""Poincare series playground.

    uv run --locked python -m apps.zeta "y^2 - x^3"
    uv run --locked python -m apps.zeta --preset cusp --upto 10
    uv run --locked python -m apps.zeta --list
    uv run --locked python -m apps.zeta "x*y" "x + y" --verify

Counts ``N_e = #{x mod 2^e : f(x) = 0}`` with Algorithm 6 on the subset-lattice
kernel, then recovers the rational generating function Igusa's theorem
guarantees exists.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from typing import List, Sequence

from padic.poly import Poly

from .parse import ParseError, parse_system
from .series import poincare

#: Classical singularities, so there is something to type at it immediately.
PRESETS = {
    "square":   (["x^2"],                  "double point: N_e = 2^floor(e/2)"),
    "cube":     (["x^3"],                  "triple point: N_e = 2^(e - ceil(e/3))"),
    "node":     (["x*y"],                  "node (two crossing lines): N_e = (e+2)2^(e-1)"),
    "cusp":     (["y^2 - x^3"],            "cusp -- the textbook Igusa example"),
    "tacnode":  (["y^2 - x^4"],            "tacnode"),
    "d4":       (["x^3 - x*y^2"],          "D4: three concurrent lines"),
    "sum2":     (["x^2 + y^2"],            "2-adically isotropic binary form"),
    "pell":     (["x^2 - 2*y^2"],          "anisotropic binary form"),
    "sum3":     (["x^2 + y^2 + z^2"],      "ternary quadratic form"),
    "fermat3":  (["x^3 + y^3 + z^3"],      "Fermat cubic surface"),
    "whitney":  (["x^2 - y^2*z"],          "Whitney umbrella"),
    "ec":       (["y^2 - x^3 - x - 1"],    "elliptic curve, good reduction at 2"),
    "sing-ec":  (["y^2 - x^3 - x^2"],      "nodal cubic (singular at the origin)"),
    "line-pair": (["x*y", "x + y"],        "a system, not just one polynomial"),
}

#: Closed forms we can check against, where one is classical and short.
CLOSED_FORM = {
    "square": lambda e: 2 ** (e // 2),
    "cube":   lambda e: 2 ** (e - -(-e // 3)),
    "node":   lambda e: (e + 2) * 2 ** (e - 1) if e else 1,
}


def brute_force(polys: Sequence[Poly], L: int, upto: int) -> List[int]:
    out = [1]
    for e in range(1, upto + 1):
        mod = 2**e
        out.append(sum(1 for z in itertools.product(range(mod), repeat=L)
                       if all(f.eval_int(z) % mod == 0 for f in polys)))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="apps.zeta", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("expr", nargs="*", help="polynomial(s) in x, y, z, ... over Z")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--list", action="store_true", help="show the presets and exit")
    ap.add_argument("--upto", type=int, default=8, help="highest e to compute")
    ap.add_argument("--backend", default="structural",
                    choices=["structural", "dense", "numpy"])
    ap.add_argument("--max-survivors", type=int, default=1 << 22)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check small e against exhaustive search")
    args = ap.parse_args(argv)

    if args.list:
        print("presets:\n")
        for name, (srcs, note) in sorted(PRESETS.items()):
            print(f"  {name:<11} {' , '.join(srcs):<22}  {note}")
        return 0

    sources = PRESETS[args.preset][0] if args.preset else args.expr
    if not sources:
        ap.error("give a polynomial, or --preset NAME, or --list")

    try:
        polys, names = parse_system(sources)
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2
    if not names:
        print("no variables in that expression", file=sys.stderr)
        return 2
    L = len(names)

    print(f"system      {' = 0,  '.join(sources)} = 0")
    print(f"variables   {', '.join(names)}   (L = {L})")
    print(f"degree      {max(f.degree for f in polys)}"
          f"      terms {sum(len(f.terms) for f in polys)}")
    if L > 24:
        print(f"L = {L} is beyond what a 2^L table fits", file=sys.stderr)
        return 2

    res = poincare(polys, L, args.upto, backend=args.backend,
                   max_survivors=args.max_survivors)

    # once the series is known, it predicts the terms the cap truncated
    pred = (res.rational.coefficients(len(res.N) - 1)
            if res.rational is not None else None)

    print(f"\nN_e = #{{x mod 2^e : f(x) = 0}}   [{res.backend}, {res.seconds:.2f}s]")
    for e, n in enumerate(res.N):
        ratio = f"   x{res.N[e]/res.N[e-1]:.3f}" if e and res.N[e - 1] else ""
        if e > res.exact_upto:
            note = "  (cap reached: a lower bound, not a count)"
            if pred is not None and pred[e].denominator == 1:
                note = f"  -> {int(pred[e]):,} from the fitted series"
            print(f"  N_{e:<2} = {n:>14,}{note}")
        else:
            print(f"  N_{e:<2} = {n:>14,}{ratio}")

    dim = res.dimension_estimate
    if dim is not None:
        print(f"\ngrowth  N_(e+1)/N_e -> 2^{dim:.2f}"
              f"   (dimension of the zero locus over Z_2)")

    if res.rational is not None:
        print(f"\nPoincare series   sum_e N_e t^e  =  {res.rational}")
        c = res.recurrence
        terms = "  ".join(f"{ci:+} N_(e-{i + 1})"
                          for i, ci in enumerate(c) if ci != 0)
        print(f"recurrence        N_e = {terms or '0'}   (order {len(c)})")
    else:
        print("\nno rational form found from these terms -- try a larger --upto")
        print("(Igusa's theorem says one exists; the fit needs enough exact terms)")

    rc = 0
    if args.verify:
        depth = {1: 12, 2: 7, 3: 5, 4: 4}.get(L, 3)
        depth = min(depth, args.upto)
        bf = brute_force(polys, L, depth)
        ok = res.N[: depth + 1] == bf
        print(f"\nverify  exhaustive search over (Z/2^e)^{L} for e <= {depth}: "
              f"{'AGREE' if ok else 'MISMATCH'}")
        if not ok:
            print(f"  dp    {res.N[: depth + 1]}\n  brute {bf}")
            rc = 1
        if args.preset in CLOSED_FORM:
            cf = [CLOSED_FORM[args.preset](e) for e in range(len(res.N))]
            ok2 = cf == res.N
            print(f"verify  closed form {PRESETS[args.preset][1]}: "
                  f"{'AGREE' if ok2 else 'MISMATCH'}")
            if not ok2:
                print(f"  dp     {res.N}\n  closed {cf}")
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
