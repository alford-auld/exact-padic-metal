"""The Poincare series playground: parser, counts, and the rational fit."""

import itertools
from fractions import Fraction

import pytest

from apps.zeta.__main__ import CLOSED_FORM, PRESETS, brute_force, main
from apps.zeta.parse import ParseError, parse, parse_system
from apps.zeta.series import fit_rational, find_recurrence, poincare
from conftest import requires_metal
from padic.poly import Poly


# -- parser ----------------------------------------------------------------


def test_parses_a_polynomial():
    f, names = parse("x^2 + 3*y - 4")
    assert names == ["x", "y"]
    assert f.eval_int([2, 5]) == 4 + 15 - 4


def test_variable_order_is_canonical_not_first_appearance():
    """`y^2 - x^3` and `x^3 - y^2` must give the same layout."""
    _, a = parse_system(["y^2 - x^3"])
    _, b = parse_system(["x^3 - y^2"])
    assert a == b == ["x", "y"]


def test_natural_ordering_of_indexed_names():
    _, names = parse_system(["x10 + x2 + x1"])
    assert names == ["x1", "x2", "x10"]


def test_precedence_and_parentheses():
    f, _ = parse("2*x^2")
    g, _ = parse("(2*x)^2")
    assert f.eval_int([3]) == 18
    assert g.eval_int([3]) == 36


def test_unary_minus_and_nested_parens():
    f, _ = parse("-(x - 2)*(x + 2)")
    assert f.eval_int([5]) == -(3) * 7


@pytest.mark.parametrize("src", ["x^", "x +", "*x", "x^-2", "x^y", "2 $ x", "(x"])
def test_malformed_input_raises_parse_error(src):
    with pytest.raises(ParseError):
        parse(src)


def test_unknown_variable_is_rejected():
    with pytest.raises(ParseError, match="unknown variable"):
        parse("x + q", names=["x"])


def test_parser_does_not_evaluate_python():
    """It is a parser, not eval: Python syntax must not execute."""
    with pytest.raises(ParseError):
        parse("__import__('os').getcwd()")


# -- counts ----------------------------------------------------------------


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_matches_exhaustive_search(preset):
    polys, names = parse_system(PRESETS[preset][0])
    L = len(names)
    depth = {1: 8, 2: 6, 3: 4}[L]
    res = poincare(polys, L, depth)
    assert res.N == brute_force(polys, L, depth)


@pytest.mark.parametrize("preset", sorted(CLOSED_FORM))
def test_presets_with_closed_forms_match_them(preset):
    polys, names = parse_system(PRESETS[preset][0])
    res = poincare(polys, len(names), 10)
    assert res.N == [CLOSED_FORM[preset](e) for e in range(11)]


def test_N_0_is_one():
    """(Z/p^0)^L is a single point and every system vanishes on it."""
    polys, names = parse_system(["x + 1"])
    assert poincare(polys, len(names), 3).N[0] == 1


def test_empty_zero_set_gives_zeros_not_a_short_list():
    polys, names = parse_system(["2*x + 1"])  # odd, never zero mod 2
    res = poincare(polys, len(names), 5)
    assert res.N == [1, 0, 0, 0, 0, 0]


@requires_metal
@pytest.mark.parametrize("preset", ["node", "cusp", "whitney"])
def test_metal_backend_gives_identical_counts(preset):
    polys, names = parse_system(PRESETS[preset][0])
    L = len(names)
    assert poincare(polys, L, 6, backend="dense").N == poincare(polys, L, 6).N


# -- the rational fit ------------------------------------------------------


def test_recovers_the_node_generating_function():
    """N_e = (e+2)2^(e-1)  <->  (1 - t)/(1 - 4t + 4t^2)."""
    polys, names = parse_system(["x*y"])
    res = poincare(polys, len(names), 10)
    assert res.recurrence == [Fraction(4), Fraction(-4)]
    assert str(res.rational) == "(1 - t) / (1 - 4*t + 4*t^2)"


def test_recovers_the_square_generating_function():
    polys, names = parse_system(["x^2"])
    res = poincare(polys, len(names), 10)
    assert str(res.rational) == "(1 + t) / (1 - 2*t^2)"


def test_fit_reproduces_every_exact_term_it_was_given():
    """Only the terms below the cap: beyond it N_e is a lower bound, not a count."""
    for preset in ("square", "cube", "node", "sum2", "sum3", "whitney", "d4"):
        polys, names = parse_system(PRESETS[preset][0])
        res = poincare(polys, len(names), 12)
        assert res.rational is not None, preset
        k = res.exact_upto
        got = [int(x) for x in res.rational.coefficients(k)]
        assert got == res.N[: k + 1], preset


def test_fit_extrapolates_past_the_survivor_cap():
    """Rationality is the point: a few exact terms predict the rest.

    Whitney at upto=12 saturates the cap, so the counts stop being counts --
    but the series fitted from the exact prefix keeps going, and it does not
    agree with the truncated values (it is larger, as a count must be).
    """
    polys, names = parse_system(PRESETS["whitney"][0])
    res = poincare(polys, len(names), 12, max_survivors=1 << 22)
    assert res.exact_upto < 12, "expected the cap to bite here"
    pred = [int(x) for x in res.rational.coefficients(12)]
    for e in range(res.exact_upto + 1, 13):
        assert pred[e] > res.N[e]


def test_dimension_comes_from_the_dominant_root():
    """A short window would read the period, not the growth."""
    polys, names = parse_system(["x^2"])  # ratios alternate 1, 2, 1, 2
    res = poincare(polys, len(names), 12)
    assert res.dimension_estimate == pytest.approx(0.5, abs=1e-9)


def test_no_short_recurrence_for_an_aperiodic_sequence():
    """Any d terms fit *some* order-d recurrence; the slack must prevent that."""
    assert fit_rational([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]) is None


def test_fibonacci_is_order_two():
    assert find_recurrence([1, 1, 2, 3, 5, 8, 13, 21, 34]) == [Fraction(1), Fraction(1)]


def test_geometric_sequence_is_order_one():
    assert find_recurrence([1, 2, 4, 8, 16, 32, 64]) == [Fraction(2)]


# -- CLI -------------------------------------------------------------------


def test_cli_verifies_a_preset(capsys):
    assert main(["--preset", "node", "--upto", "6", "--verify"]) == 0
    out = capsys.readouterr().out
    assert "AGREE" in out and "Poincare series" in out


def test_cli_lists_presets(capsys):
    assert main(["--list"]) == 0
    assert "cusp" in capsys.readouterr().out


def test_cli_rejects_a_bad_expression(capsys):
    assert main(["x^^2"]) == 2
