# Deviations from the printed pseudocode

Source: T. Mihara, *p-adic Character Neural Network*, [arXiv:2603.29905v1](https://arxiv.org/abs/2603.29905v1) [math.NT], 31 Mar 2026.

Every deviation below is implemented in `padic/reference.py` with a comment at
the site, and each is covered by a test that pins the corrected behaviour to an
independent ground truth. Algorithms 1, 2, 3 and 6 are transcribed verbatim and
are not listed.

## 1. Algorithm 5 does not compute `a^x`

**Printed:**

```
y <- 0
while x != 0 do
  if x mod 2 = 1 then y <- y x mod p^E
  x <- x^2 mod p^E
  x <- floor(x/2)
```

Three faults, all consistent with the base variable and the exponent variable
having been conflated into `x`:

- the accumulator starts at `y <- 0`, so the result is `0` for every input;
- the argument `a` is never read;
- the *exponent* is squared (`x <- x^2`) and then halved, rather than a running
  base being squared while the exponent is halved.

**Implemented:** the classical binary modular method the surrounding text names,
which is what its complexity claim `O((log_2 x) t(p^E))` describes.

```
y <- 1;  b <- a mod p^E
while x != 0 do
  if x mod 2 = 1 then y <- y b mod p^E
  b <- b^2 mod p^E
  x <- floor(x/2)
```

**Check:** `tests/test_network.py::test_character_equals_a_to_the_x` and the
three-way agreement in `test_taylor_polynomial_agrees_with_algorithm_4`.

## 2. Algorithm 4 truncates the running precision too early

**Printed** (line 14): `e' <- max{0, E - e m}`, with `u` and `b` then carried
modulo `p^(e')`.

The term contributed at index `e` is `p^(v + m e) u b` where `v = -v_p(e!)`.
It is significant modulo `p^E` whenever `v + m e < E`, so `u b` is needed to
precision `E - (v + m e) = E - m e + v_p(e!)`. That exceeds the printed
`E - m e` by exactly `v_p(e!)`.

Concretely at `p = 2`, `E = 10`, `e = 5`: the printed bound gives `e' = 0`,
which forces `u = 0` and drops the term — but `q^5/5! = 2^10/120` has valuation
7 and still contributes modulo `2^10`.

A first attempt to fix this by stopping the loop once the bound reaches zero is
**also wrong**, and this is the subtler point: `v + m e = m e - (e - s_p(e))/(p-1)`
is not monotonic in `e`. At `p = 2` it is `e + s_2(e)`, which *dips* at every
power of two — `e = 7` gives 10, `e = 8` gives 9. A term can therefore fall back
below `p^E` after an earlier one has exceeded it.

**Implemented:** the running state `u`, `b` is carried at full precision `p^E`,
and only the decision to add a term uses `v + m e < E`. This changes none of the
paper's complexity claims — `u` and `b` were already elements of a residue ring
of size at most `p^E`.

The truncation bound `E' = ceil(E / (m - 1/(p-1)))` on the *number of terms* is
correct as printed and is used unchanged; it is sound because
`v + m e >= e (m - 1/(p-1))` by Legendre's formula.

**Check:** `tests/test_network.py::test_taylor_polynomial_agrees_with_algorithm_4`
against `exp_p` summed exactly in `fractions.Fraction`, for `p in {2,3,5}` and
`E <= 11`.

## 3. Algorithm 6 needs a level bound to terminate

Not a fault — the paper states it: the printed `while True` never halts when
`f` has a common zero in `Z_p`, and "it suffices to break the while loop when
`e = E`". We take that bound as a required argument (`max_e`) rather than an
option, and report `guarantee = "lower_bound"` whenever it is what stopped the
loop. See `docs/exactness.md`.

## 4. Reading `|p|^(e_max - 1)` in §4

§4 states that when some `e <= E` admits no solution, the minimum of the
objective is `|p|^(e_max - 1)`, "where `e_max` denotes the maximum of such an
`e`". Read literally — the maximum `e` with *no* solution — this would always be
`E`, since insolubility at `e` implies insolubility above it.

The consistent reading, and the one matching Algorithm 6's own caption ("the
maximum of an `e` for which ... has a common zero"), is that the displayed
`e_max` is the *minimal* level with no solution, i.e. `e* + 1` where `e*` is
Algorithm 6's return value. Then `|p|^(e_max - 1) = |p|^(e*)`, which is also
what the direct argument gives: some `z` achieves valuation `e*` and none
achieves `e* + 1`.

**Implemented:** `NetworkInstance.objective_valuation(e_star)` returns
`e* - F`, the exponent for the *original* (unscaled) objective, undoing the
`p^F` that cleared the denominators of `C`.
