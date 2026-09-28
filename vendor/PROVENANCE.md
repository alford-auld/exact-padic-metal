# Vendored: the Yates butterfly kernel

`vendor/yates/` is a verbatim copy of the `yates/` package from
**exact-yates-metal** (`git@github-personal:alford-auld/exact-yates-metal.git`),
at commit:

    2f0823d585a5eb0e4b71ed368af016defa7b8d36
    f(10) = 17: the growth of max v2(c_chi) crosses 64 before n=29

MIT licensed; the upstream licence is kept alongside as
`LICENSE.exact-yates-metal`.

## Why vendored rather than a submodule

Every run executes an immutable source snapshot of the recorded commit, and
`git archive`-style snapshots **do not include submodule contents**. A
submodule therefore produced a run that failed at import with
`ImportError: the Yates kernel is not at .../vendor/exact-yates-metal`.
Copying the package in keeps the run recipe reproducible from the committed
tree alone, on any backend, with no network access and no credentials.

## Updating

Copy `yates/` from a fresh checkout, update the commit above, and re-run
`uv run --locked python -m pytest` — `tests/test_exactness.py` checks the
transform bitwise against an independent CPU implementation, so a bad copy
fails loudly.
