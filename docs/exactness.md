# The exactness contract

**Read this before quoting an `e_max`.** It says what the number means and what
it does not.

## Arithmetic: wraparound is not an approximation

Algorithm 6 only ever asks a yes/no question: does `f_i(w)` vanish modulo
`2^(e+1)`? The Yates kernel computes in `uint32`/`uint64`, where overflow in
the Metal Shading Language is defined wraparound, i.e. exact reduction modulo
`2^32` / `2^64`.

Since

    Z/2^64  ->>  Z/2^(e+1)      is a ring surjection for every  e + 1 <= 64,

reducing the kernel's `uint64` output modulo `2^(e+1)` gives *the* answer, not
an approximation to it. There is no error term to bound and no headroom to
reserve. Integer coefficients of any size are reduced modulo `2^64` on the way
in; `tests/test_exactness.py::test_overflowing_coefficients_still_give_exact_residues`
pins this with coefficients above `2^200`.

Two consequences:

- `ZETA_SUB` is **unipotent** (determinant 1, Smith normal form `diag(1,1)`),
  so `yates.bit_loss("ZETA_SUB", n) == 0` at every `n` and the transform is an
  exact automorphism of `Z/2^64`. Unlike the Hadamard case, nothing is lost.
- `max_e > 64` is **refused**, because above that the surjection argument stops
  applying. Supporting it would need multi-limb arithmetic.

The transform itself is checked bitwise against an independent CPU
implementation at every size the solver uses.

## Result: one-sided when capped

`solve()` returns a `Result` carrying a `guarantee`, and it is the field to
read first.

| `guarantee`    | meaning |
|----------------|---------|
| `"exact"`      | the dynamic program ran to a genuinely empty `W`. `e_max` is the true maximum: a common zero exists modulo `2^e_max` and none exists modulo `2^(e_max+1)`. |
| `"lower_bound"` | the program was stopped early. A common zero modulo `2^e_max` **does** exist; nothing is claimed above it. |

Exactly two things produce `"lower_bound"`, and both can only make the answer
*too small* — never too large:

1. **The level bound `max_e`.** The paper's own remedy for the `while True`
   that never halts when a zero exists in `Z_p`. Hitting it means zeros were
   still alive when we stopped.
2. **The survivor cap `max_survivors`.** Each surviving `z` lifts to up to
   `2^(L - rank)` successors, so `|Z|` can multiply every level (see the
   `BLOWUP` section of `bench/run.py` for measured growth). Discarding
   survivors can only remove candidates that might have lifted further, so the
   program can stop earlier than it should — never later.

The asymmetry is the whole content: a reported `e_max` is always *achievable*,
and `Result.levels` records which levels truncated, so a capped run can be
re-run with a larger cap to tighten it.

## What `e_max` means for the network

For a `NetworkInstance`, the scaled system is solved at precision
`E* = E + F`. If `e*` is the returned level, the minimum of the scaled
objective is `|p|^(e*)` and the minimum of the **original** objective is
`|p|^(e* - F)` — `NetworkInstance.objective_valuation`. When `e* = E*` the
network fits to the full precision the data was observed at, and the residual
is zero as far as the hypotheses of §4 can tell; the underlying problem may or
may not have an exact solution in `Z_p`, and this computation does not decide
that.
