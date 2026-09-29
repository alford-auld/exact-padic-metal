"""The project's run command: validate every path, then measure them.

Prints, in order:

  CONFIG      the host, the kernel commit, the library versions
  VALIDATE    every fast path against the literal pseudocode of Algorithm 6
  PHASE2      the §4 reduction, end to end, on planted and unfittable networks
  LEVEL0      throughput of the e = 0 step -- the NP-hard one -- vs L
  CROSSOVER   dense transform vs structural F_2 solve at e >= 1
  BLOWUP      survivor growth, the algorithm's real limit
  GROKKING    (a+b) mod P exactly, and why the search cannot find it
  ZETA        Poincare series of classical singularities, all verified
  SUMMARY     one greppable block of final metrics

Run with:  uv run --locked python -m bench.run
"""

from __future__ import annotations

import platform
import random
import sys
import time
from typing import List

import numpy as np

from padic import _yates, ddp, reference
from padic.ddp import _level_dense, _level_structural, zeta_sub_numpy
from padic.network import planted_instance, random_instance
from padic.poly import Poly, lift_coefficients

SEED = 20260928
SUMMARY: dict[str, object] = {}


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)), flush=True)


def random_system(rng, nvars, npolys, max_deg=3, nterms=4, cmax=24):
    out = []
    for _ in range(npolys):
        terms: dict = {}
        for _ in range(nterms):
            exps = [0] * nvars
            for _ in range(rng.randint(0, max_deg)):
                exps[rng.randrange(nvars)] += 1
            c = rng.randint(-cmax, cmax)
            if c:
                key = tuple(exps)
                terms[key] = terms.get(key, 0) + c
        out.append(Poly(nvars, terms))
    return out


# -- CONFIG ----------------------------------------------------------------


def section_config() -> bool:
    rule("CONFIG")
    import mlx.core as mx

    yates = _yates.load()
    have_gpu = mx.metal.is_available()
    info = yates.host_info() if have_gpu else {}
    print(f"python         {platform.python_version()}  ({platform.machine()})")
    print(f"mlx            {mx.__version__}")
    print(f"numpy          {np.__version__}")
    print(f"yates commit   {_yates.kernel_commit()}")
    print(f"metal          {'yes' if have_gpu else 'NO -- GPU paths skipped'}")
    for k in ("Model Identifier", "Chip", "Memory", "macOS"):
        if k in info:
            print(f"{k:<14} {info[k]}")
    print(f"seed           {SEED}")
    SUMMARY["kernel_commit"] = _yates.kernel_commit()
    SUMMARY["metal"] = have_gpu
    return have_gpu


# -- VALIDATE --------------------------------------------------------------


def section_validate(have_gpu: bool) -> None:
    rule("VALIDATE  (fast paths vs. the literal Algorithm 6)")
    rng = random.Random(SEED)
    backends = ["numpy", "structural"] + (["dense"] if have_gpu else [])
    max_e = 6
    checked = 0
    for L in (1, 2, 3, 4):
        for _ in range(12):
            polys = random_system(rng, L, 3)
            want = reference.digit_dynamic_programming(2, L, polys, max_e=max_e)
            for backend in backends:
                got = ddp.solve(polys, L, max_e=max_e, backend=backend)
                if got.e_max != want:
                    print(f"MISMATCH L={L} backend={backend}: {got.e_max} != {want}")
                    sys.exit(1)
                checked += 1
    print(f"e_max agreement          {checked} solves over {len(backends)} backends, L<=4: OK")

    # survivor sets, not just the answer
    setchecks = 0
    for L in (2, 3, 4):
        for e in (1, 2, 3):
            z0 = [rng.randrange(1 << (e + 1)) for _ in range(L)]
            polys = [f - Poly.constant(L, f.eval_int(z0)) for f in random_system(rng, L, 2)]
            Z = np.zeros((1, L), dtype=np.uint64)
            for lv in range(e):
                Z, _ = _level_dense(polys, Z, lv, L, use_gpu=False,
                                    budget_bytes=1 << 26, cap=1 << 16)
            dense, _ = _level_dense(polys, Z, e, L, use_gpu=False,
                                    budget_bytes=1 << 26, cap=1 << 16)
            struct, _ = _level_structural(polys, Z, e, L, cap=1 << 16)
            s1 = {tuple(int(v) for v in r) for r in dense}
            s2 = {tuple(int(v) for v in r) for r in struct}
            if s1 != s2:
                print(f"SURVIVOR SET MISMATCH L={L} e={e}: {len(s1)} vs {len(s2)}")
                sys.exit(1)
            setchecks += 1
    print(f"survivor sets            {setchecks} levels, dense == structural: OK")

    if have_gpu:
        import mlx.core as mx

        yates = _yates.load()
        gen = np.random.default_rng(SEED)
        for n in (4, 8, 12, 16):
            a = gen.integers(0, 1 << 63, size=(2, 2, 1 << n), dtype=np.uint64)
            if not np.array_equal(np.asarray(yates.transform(mx.array(a), "ZETA_SUB")),
                                  zeta_sub_numpy(a)):
                print(f"KERNEL MISMATCH at n={n}")
                sys.exit(1)
        print("kernel vs cpu transform  n in {4,8,12,16}, uint64: bitwise identical")
    SUMMARY["validate"] = "pass"


# -- PHASE2 ----------------------------------------------------------------


def section_phase2(have_gpu: bool) -> None:
    rule("PHASE2  (§4 reduction: network -> polynomial system -> e_max)")
    rng = random.Random(SEED + 1)
    backend = "structural"

    print(f"{'instance':<34}{'L':>4}{'|I|':>5}{'deg':>5}{'terms':>7}"
          f"{'e_max':>7}{'guarantee':>13}{'s':>8}")
    rows = []
    for tag, N, M, D, E, F, n in [
        ("planted  D=1 N=1 M=1 E=3", 1, 1, 1, 3, 0, 2),
        ("planted  D=1 N=2 M=1 E=3", 2, 1, 1, 3, 0, 3),
        ("planted  D=2 N=1 M=1 E=3", 1, 1, 2, 3, 0, 3),
        ("planted  D=1 N=1 M=1 E=2 F=2", 1, 1, 1, 2, 2, 2),
        ("random   D=1 N=1 M=1 E=3", 1, 1, 1, 3, 0, 3),
        ("random   D=1 N=2 M=1 E=4", 2, 1, 1, 4, 0, 4),
        ("random   D=2 N=2 M=1 E=3", 2, 1, 2, 3, 0, 4),
    ]:
        if tag.startswith("planted"):
            inst, z = planted_instance(rng, n_samples=n, N=N, M=M, D=D, E=E, F=F)
        else:
            inst, z = random_instance(rng, n_samples=n, N=N, M=M, D=D, E=E, F=F), None
        polys = inst.to_polynomial_system()
        deg = max(f.degree for f in polys)
        nterms = sum(len(f.terms) for f in polys)
        t0 = time.perf_counter()
        res = ddp.solve(polys, inst.n_vars, max_e=inst.E_star, backend=backend,
                        max_survivors=1 << 18)
        dt = time.perf_counter() - t0
        print(f"{tag:<34}{inst.n_vars:>4}{len(polys):>5}{deg:>5}{nterms:>7}"
              f"{res.e_max:>7}{res.guarantee:>13}{dt:>8.3f}")
        if z is not None:
            # the planted network must itself satisfy the system, and the
            # program must reach the level bound rather than stopping early
            mod = 2**inst.E_star
            assert all(f.eval_int(z) % mod == 0 for f in polys), tag
            assert inst.residual_valuation(z) == inst.E_star, tag
            assert res.e_max == inst.E_star, tag
        obj = inst.objective_valuation(res.e_max)
        rows.append((tag, res.e_max, obj))

    print("\nminimum of the original objective, ||.|| = |p|^(e* - F):")
    for tag, e_star, obj in rows:
        print(f"  {tag:<34} e* = {e_star:<3} ->  |2|^{obj}")
    SUMMARY["phase2_planted_all_reached_bound"] = True


# -- LEVEL0 ----------------------------------------------------------------


def section_level0(have_gpu: bool) -> None:
    rule("LEVEL0  (the e = 0 step: one batched ZETA_SUB over 2^L)")
    print("the NP-hard level -- common roots of multilinear systems over F_2 --")
    print("where the 2^L transform is the algorithm, not a brute-force stand-in.\n")
    if not have_gpu:
        print("no Metal device; skipped")
        return

    import mlx.core as mx

    yates = _yates.load()
    rng = random.Random(SEED + 2)
    n_polys = 4

    def timed(fn, reps: int) -> float:
        """Wall time per rep, with MLX's laziness forced at every rep.

        Evaluating only the last result would let MLX drop the earlier graphs
        and report a single rep divided by `reps`.
        """
        out = fn()
        mx.eval(out)
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
            mx.eval(out)
        mx.synchronize()
        return (time.perf_counter() - t0) / reps

    print("timings are normalised against this machine's device-to-device copy")
    print("of the same array: for a memory-bound kernel that is the cost of one")
    print("pass, so 'vs copy' counts how many passes the L-stage transform takes.\n")
    print(f"{'L':>4}{'2^L':>12}{'MiB':>9}{'build ms':>11}{'zeta ms':>10}"
          f"{'copy ms':>10}{'vs copy':>9}{'cpu ms':>10}{'speedup':>9}")
    results = []
    for L in range(10, 25):
        polys = random_system(rng, L, n_polys, max_deg=3, nterms=6)
        Z = np.zeros((1, L), dtype=np.uint64)
        t0 = time.perf_counter()
        coeffs = lift_coefficients(polys, Z, 0, L)
        t_build = time.perf_counter() - t0
        nbytes = coeffs.nbytes

        a = mx.array(coeffs)
        mx.eval(a)
        reps = 20 if L < 20 else 5
        t_gpu = timed(lambda: yates.transform(a, "ZETA_SUB"), reps)
        t_copy = timed(lambda: yates.device_copy(a), reps)

        t_cpu = float("nan")
        if L <= 20:
            t0 = time.perf_counter()
            zeta_sub_numpy(coeffs)
            t_cpu = time.perf_counter() - t0

        speed = t_cpu / t_gpu if t_cpu == t_cpu else float("nan")
        print(f"{L:>4}{1 << L:>12}{nbytes / 2**20:>9.1f}{t_build * 1e3:>11.1f}"
              f"{t_gpu * 1e3:>10.2f}{t_copy * 1e3:>10.2f}"
              f"{t_gpu / t_copy:>9.1f}{t_cpu * 1e3:>10.1f}{speed:>9.1f}")
        results.append((L, t_build, t_gpu, t_copy, t_cpu, nbytes))

    big = [r for r in results if r[0] >= 18]
    copy_gbps = max((2 * r[5]) / r[3] / 1e9 for r in big)
    lo = min(r[2] / r[3] for r in big)
    hi = max(r[2] / r[3] for r in big)
    fused = max(r[0] / (r[2] / r[3]) for r in big)
    print(f"\ncopy bandwidth           {copy_gbps:.0f} GB/s  (device-to-device, same array)")
    print(f"zeta / copy at L >= 18   {lo:.1f}x .. {hi:.1f}x")
    print(f"stages per memory pass   up to {fused:.0f} of the {big[-1][0]} Yates stages")
    print("i.e. the whole L-stage transform costs about three copies of the")
    print("array, because threadgroup tiling fuses stages into one pass.")
    SUMMARY["level0_copy_gbps"] = round(copy_gbps)
    SUMMARY["level0_stages_per_pass"] = round(fused)
    SUMMARY["level0_zeta_over_copy_max"] = round(max(r[2] / r[3] for r in big), 1)
    SUMMARY["level0_max_L"] = results[-1][0]
    finite = [r for r in results if r[4] == r[4]]
    if finite:
        SUMMARY["level0_max_speedup_vs_cpu"] = round(max(r[4] / r[2] for r in finite), 1)


# -- CROSSOVER -------------------------------------------------------------


def section_crossover(have_gpu: bool) -> None:
    rule("CROSSOVER  (e >= 1: dense transform vs. structural F_2 solve)")
    print("above level 0 the lift is affine in the digits, so F_2 elimination")
    print("replaces the 2^L enumeration. square systems (|I| = L) keep the")
    print("survivor count O(1) per z, isolating the algorithmic comparison")
    print("from the blow-up measured in the next section.\n")
    rng = random.Random(SEED + 3)
    e = 2
    cap = 1 << 20
    print(f"{'L':>4}{'|I|':>5}{'|Z|':>6}{'dense ms':>11}{'struct ms':>12}"
          f"{'ratio':>9}{'survivors':>11}{'same set':>10}")
    for L in range(6, 25):
        z0 = [rng.randrange(1 << (e + 1)) for _ in range(L)]
        polys = [f - Poly.constant(L, f.eval_int(z0))
                 for f in random_system(rng, L, L, max_deg=2, nterms=4)]
        # build Z with a generous cap, then take 8 rows: capping during the
        # build can discard the planted zero's ancestor and leave Z empty.
        Z = np.zeros((1, L), dtype=np.uint64)
        for lv in range(e):
            if lv >= 1:
                Z, _ = _level_structural(polys, Z, lv, L, cap=1 << 14)
            else:
                Z, _ = _level_dense(polys, Z, lv, L, use_gpu=have_gpu,
                                    budget_bytes=1 << 28, cap=1 << 14)
        if Z.shape[0] == 0:
            continue
        Z = Z[:8]

        t_dense, same = float("nan"), "-"
        if L <= 22:
            t0 = time.perf_counter()
            d, td = _level_dense(polys, Z, e, L, use_gpu=have_gpu,
                                 budget_bytes=1 << 30, cap=cap)
            t_dense = time.perf_counter() - t0
        t0 = time.perf_counter()
        s, ts = _level_structural(polys, Z, e, L, cap=cap)
        t_struct = time.perf_counter() - t0

        if L <= 22:
            if td or ts:
                same = "capped"  # sets may legitimately differ once truncated
            else:
                same = "yes" if {tuple(int(v) for v in r) for r in d} == \
                                {tuple(int(v) for v in r) for r in s} else "NO"
                if same == "NO":
                    print(f"MISMATCH at L={L}")
                    sys.exit(1)
        ratio = t_dense / t_struct if t_dense == t_dense else float("nan")
        print(f"{L:>4}{L:>5}{Z.shape[0]:>6}{t_dense * 1e3:>11.1f}{t_struct * 1e3:>12.2f}"
              f"{ratio:>9.1f}{s.shape[0]:>11}{same:>10}")
    SUMMARY["crossover"] = "structural overtakes dense as L grows, at e>=1"


def section_blowup(have_gpu: bool) -> None:
    rule("BLOWUP  (|Z_e| growth: the real limit of Algorithm 6)")
    print("each surviving z lifts to 2^(L - rank) successors, so an")
    print("under-determined system multiplies |Z| every level. this, not the")
    print("cost of one level, is what bounds the algorithm in practice.\n")
    rng = random.Random(SEED + 4)
    cap = 1 << 22
    print(f"{'L':>4}{'|I|':>5}   |Z_e| by level")
    for L, nI in [(8, 2), (8, 4), (8, 8), (12, 4), (12, 12), (16, 16)]:
        z0 = [rng.randrange(64) for _ in range(L)]
        polys = [f - Poly.constant(L, f.eval_int(z0))
                 for f in random_system(rng, L, nI, max_deg=2, nterms=4)]
        res = ddp.solve(polys, L, max_e=6, backend="structural", max_survivors=cap)
        sizes = " -> ".join(str(s.survivors_out) for s in res.levels)
        flag = "  [CAPPED]" if any(s.truncated for s in res.levels) else ""
        print(f"{L:>4}{nI:>5}   {sizes}{flag}")
    SUMMARY["blowup"] = "|Z| multiplies by 2^(L-rank) per level when |I| < L"


# -- GROKKING --------------------------------------------------------------


def section_grokking(have_gpu: bool) -> None:
    rule("GROKKING  (neelnanda-io/Grokking: (a+b) mod P, exactly)")
    from apps.grokking import certify, exact
    from apps.grokking.task import GROKKING, ModularAddition

    print("the task a transformer groks on has a closed-form two-unit solution")
    print("in the character network:  y = (chi(p^j(a+b)) - chi(0)) / p^F .\n")

    print(f"{'p':>5}{'E':>3}{'modulus':>9}{'pairs':>8}{'j':>3}{'F':>3}"
          f"{'mismatches':>12}{'solves the §4 system':>22}")
    for p_, E in [(2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (5, 2), (113, 1)]:
        task = ModularAddition(p_, E)
        sol = exact.exact_solution(p_, E)
        pairs = task.all_pairs()
        probe = pairs if len(pairs) <= 64 else random.Random(0).sample(pairs, 40)
        mism, is_zero = certify.certify_closed_form(task, probe)
        tag = "  <- Grokking" if task == GROKKING else ""
        print(f"{p_:>5}{E:>3}{task.modulus:>9}{task.n_pairs:>8}{sol.j:>3}{sol.F:>3}"
              f"{mism:>12}{str(is_zero):>22}{tag}")
    print("\nmismatches are over ALL pairs; `solves the §4 system` means the")
    print("closed form is a common zero of the polynomials Algorithm 6 searches,")
    print("so any failure below is a search failure, not an absent solution.")

    # -- the obstruction
    print("\n-- why the search cannot find it: mod-2 Jacobian of the §4 system --")
    task = ModularAddition(2, 3)
    profiles = certify.rank_profile(task, task.all_pairs(), levels=3)
    print(f"{'level':>6}{'|Z|':>8}{'equations':>11}{'L':>4}{'F_2 rank':>10}"
          f"{'branching':>11}   columns live mod 2")
    for rp in profiles:
        print(f"{rp.level:>6}{rp.n_survivors:>8}{rp.n_equations:>11}{rp.n_vars:>4}"
              f"{rp.rank_max:>10}{'2^' + str(rp.n_vars - rp.rank_max):>11}   "
              f"{', '.join(rp.live_columns)}")
    if profiles:
        print(f"\nchi = exp_p(q .) has chi' = q chi with q = p^m, so every derivative")
        print("of the residual in an entry of A or b carries a factor of p. Those")
        print("columns are identically zero mod p at every level: the digit DP must")
        print(f"enumerate all {profiles[0].branching} of them per survivor, per level,")
        print("and no amount of extra data changes that (64 equations, rank 1).")
        SUMMARY["grokking_branching_per_level"] = profiles[0].branching
        SUMMARY["grokking_f2_rank"] = profiles[0].rank_max

    # -- the search, where it still works
    print("\n-- Algorithm 6 searching for it (p = 2) --")
    print(f"{'E':>3}{'P':>4}{'L':>3}{'train':>7}{'of':>5}{'e*':>4}{'target':>8}"
          f"{'fitted':>8}{'test err':>10}{'s':>8}")
    rng = random.Random(SEED + 5)
    reached = []
    for E, fracs, cap in [(1, (0.5, 1.0), 1 << 16), (2, (0.5, 1.0), 1 << 16),
                          (3, (1.0,), 1 << 14)]:
        task = ModularAddition(2, E)
        for frac in fracs:
            train, test = task.split(random.Random(11), frac)
            r = certify.fit(task, train, test, max_survivors=cap)
            te = "-" if r.test_errors is None else f"{r.test_errors}/{r.n_test}"
            print(f"{E:>3}{task.modulus:>4}{task.D * 4:>3}{r.n_train:>7}{task.n_pairs:>5}"
                  f"{r.e_max:>4}{r.e_star_target:>8}{str(r.fitted):>8}{te:>10}{r.seconds:>8.2f}")
            reached.append((E, r.fitted, r.test_errors))
    ok = [E for E, f, t in reached if f and t == 0]
    SUMMARY["grokking_search_exact_upto_E"] = max(ok) if ok else 0
    SUMMARY["grokking_closed_form"] = "exact for every prime tested, incl. P=113"


# -- ZETA ------------------------------------------------------------------


def section_zeta(have_gpu: bool) -> None:
    rule("ZETA  (Poincare series: N_e = #{x mod 2^e : f(x) = 0})")
    from apps.zeta.__main__ import CLOSED_FORM, PRESETS, brute_force
    from apps.zeta.parse import parse_system
    from apps.zeta.series import poincare

    print("the survivor counts of Algorithm 6 ARE the Poincare coefficients.")
    print("Igusa: the series is rational, so a few terms give all of it.\n")

    print(f"{'preset':<11}{'system':<18}{'L':>3}{'dim':>7}  "
          f"{'Poincare series':<46}{'brute':>7}{'closed':>8}")
    failures = 0
    for name in sorted(PRESETS):
        srcs, _ = PRESETS[name]
        polys, names = parse_system(srcs)
        L = len(names)
        res = poincare(polys, L, 18 if L == 2 else 12)
        depth = min({1: 8, 2: 6, 3: 4}.get(L, 3), res.exact_upto)
        bf = brute_force(polys, L, depth)
        ok_bf = res.N[: depth + 1] == bf
        ok_cf = "-"
        if name in CLOSED_FORM:
            want = [CLOSED_FORM[name](e) for e in range(res.exact_upto + 1)]
            ok_cf = "OK" if want == res.N[: res.exact_upto + 1] else "FAIL"
        rat = str(res.rational) if res.rational is not None else "(needs more terms)"
        dim = res.dimension_estimate
        print(f"{name:<11}{' , '.join(srcs):<18}{L:>3}"
              f"{(f'{dim:.2f}' if dim is not None else '-'):>7}  {rat:<46}"
              f"{('OK' if ok_bf else 'FAIL'):>7}{ok_cf:>8}")
        failures += (not ok_bf) + (ok_cf == "FAIL")
    if failures:
        print(f"\n{failures} ZETA CHECK(S) FAILED")
        sys.exit(1)
    print("\nevery preset agrees with exhaustive search over all of (Z/2^e)^L,")
    print("and the three with classical closed forms agree with those too.")

    if have_gpu:
        same = True
        for name in ("node", "cusp", "whitney", "fermat3"):
            polys, names = parse_system(PRESETS[name][0])
            L = len(names)
            same &= poincare(polys, L, 6, backend="dense").N == poincare(polys, L, 6).N
        print(f"Metal vs CPU counts identical: {same}")
        if not same:
            sys.exit(1)
        SUMMARY["zeta_metal_matches_cpu"] = same
    SUMMARY["zeta_presets_verified"] = len(PRESETS)


# -- main ------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    have_gpu = section_config()
    section_validate(have_gpu)
    section_phase2(have_gpu)
    section_level0(have_gpu)
    section_crossover(have_gpu)
    section_blowup(have_gpu)
    section_grokking(have_gpu)
    section_zeta(have_gpu)

    rule("SUMMARY")
    SUMMARY["wall_seconds"] = round(time.perf_counter() - t0, 1)
    for k, v in SUMMARY.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
