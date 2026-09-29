"""A small polynomial parser, so the playground takes typed-in input.

    x^2 + y^3 - 2*x*y        ->  Poly over Z in variables x, y

Recursive descent, no ``eval``: this ships in a public repository and
evaluating user strings there would be indefensible however local the CLI is.

Grammar::

    expr   := term (('+' | '-') term)*
    term   := power ('*' power)*
    power  := atom ('^' integer)?
    atom   := integer | name | '-' atom | '(' expr ')'

Multiplication must be written out (``2*x``, not ``2x``): with multi-character
variable names implicit juxtaposition is ambiguous, and a playground that
silently reads ``xy`` as ``x*y`` when ``xy`` is also a legal name is worse than
one that asks for a star.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from padic.poly import Poly

_TOKEN = re.compile(r"\s*(\d+|[A-Za-z_][A-Za-z_0-9]*|\*\*|[-+*^()])")


class ParseError(ValueError):
    """Raised with the offending position, so the CLI can point at it."""


def tokenize(src: str) -> List[str]:
    out, pos = [], 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            if src[pos:].strip() == "":
                break
            raise ParseError(f"unexpected character {src[pos]!r} at position {pos}")
        out.append(m.group(1))
        pos = m.end()
    return out


def variables_in(src: str) -> List[str]:
    """Variable names in the order they first appear -- fixes the layout."""
    seen: List[str] = []
    for tok in tokenize(src):
        if tok[0].isalpha() or tok[0] == "_":
            if tok not in seen:
                seen.append(tok)
    return seen


class _Parser:
    def __init__(self, tokens: Sequence[str], names: Sequence[str]):
        self.toks = list(tokens)
        self.i = 0
        self.names = list(names)
        self.L = len(names)

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of expression")
        self.i += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.take()
        if got != tok:
            raise ParseError(f"expected {tok!r}, found {got!r}")

    def parse(self) -> Poly:
        out = self.expr()
        if self.peek() is not None:
            raise ParseError(f"trailing input at {self.peek()!r}")
        return out

    def expr(self) -> Poly:
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            rhs = self.term()
            node = node + rhs if op == "+" else node - rhs
        return node

    def term(self) -> Poly:
        node = self.power()
        while self.peek() == "*":
            self.take()
            node = node * self.power()
        return node

    def power(self) -> Poly:
        base = self.atom()
        if self.peek() in ("^", "**"):
            self.take()
            neg = False
            if self.peek() == "-":
                self.take()
                neg = True
            exp_tok = self.take()
            if not exp_tok.isdigit():
                raise ParseError(f"exponent must be a non-negative integer, got {exp_tok!r}")
            if neg:
                raise ParseError("negative exponents are not polynomials")
            return base ** int(exp_tok)
        return base

    def atom(self) -> Poly:
        tok = self.take()
        if tok == "(":
            node = self.expr()
            self.expect(")")
            return node
        if tok == "-":
            return -self.atom()
        if tok == "+":
            return self.atom()
        if tok.isdigit():
            return Poly.constant(self.L, int(tok))
        if tok[0].isalpha() or tok[0] == "_":
            if tok not in self.names:
                raise ParseError(f"unknown variable {tok!r}; known: {self.names}")
            return Poly.var(self.L, self.names.index(tok))
        raise ParseError(f"unexpected token {tok!r}")


def parse(src: str, names: Sequence[str] | None = None) -> tuple[Poly, List[str]]:
    """Parse one polynomial.  Returns it and the variable names it uses."""
    names = list(names) if names is not None else variables_in(src)
    return _Parser(tokenize(src), names).parse(), names


def _natural_key(name: str):
    """x2 before x10, and x before y -- a stable, unsurprising ordering."""
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r"(\d+)", name))


def parse_system(sources: Sequence[str]) -> tuple[List[Poly], List[str]]:
    """Parse several polynomials into one canonical variable layout.

    The layout is sorted rather than first-appearance, so ``y^2 - x^3`` and
    ``x^3 - y^2`` give the same variable indices and the same output.
    """
    names: List[str] = []
    for src in sources:
        for n in variables_in(src):
            if n not in names:
                names.append(n)
    names.sort(key=_natural_key)
    return [parse(src, names)[0] for src in sources], names
