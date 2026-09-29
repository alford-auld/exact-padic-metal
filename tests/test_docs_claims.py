"""Every number the READMEs quote must match the committed results artifact.

The failure this prevents: a figure is measured once, pasted into prose, the
code changes, and the prose keeps asserting the old number with no way for a
reader to tell. So the READMEs cite exactly one results file, and these tests
re-derive each published figure from it and compare at the printed precision.

Re-running `bench.run` writes `bench/results/latest.json`, which is gitignored
and which nothing cites. Regenerating the *pinned* file is therefore a
deliberate act, and it breaks these tests until the prose is updated to match
-- which is the point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINNED = ROOT / "bench" / "results" / "m4-16gb-2026-09-28.json"


@pytest.fixture(scope="module")
def results():
    if not PINNED.is_file():
        pytest.fail(f"the results file the READMEs cite is missing: {PINNED}")
    return json.loads(PINNED.read_text())


@pytest.fixture(scope="module")
def readme():
    return (ROOT / "README.md").read_text()


@pytest.fixture(scope="module")
def zeta_readme():
    return (ROOT / "apps" / "zeta" / "README.md").read_text()


def tables(md: str):
    """Every markdown table row, as a list of stripped cells."""
    rows = []
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|") and set(line) != set("|-: "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
    return rows


def num(cell: str) -> float:
    """First number in a cell like '1.32 ms', '2800×', '8 MiB'."""
    m = re.search(r"-?\d+(?:\.\d+)?", cell.replace(",", ""))
    if m is None:
        raise AssertionError(f"no number in cell {cell!r}")
    return float(m.group())


def decimals(cell: str) -> int:
    m = re.search(r"-?\d+(?:\.(\d+))?", cell.replace(",", ""))
    return len(m.group(1)) if m and m.group(1) else 0


def agrees(cell: str, value: float, what: str) -> None:
    """Compare at the precision the README actually prints.

    A fixed tolerance is the wrong rule here: '5×' and '1.32 ms' assert
    different things about the underlying measurement, and the document should
    be held to what it claims rather than to an arbitrary epsilon.
    """
    d = decimals(cell)
    assert num(cell) == round(value, d) if d else num(cell) == round(value), (
        f"{what}: README prints {cell!r}, results file gives "
        f"{round(value, d) if d else round(value)}")


# -- provenance ------------------------------------------------------------


def test_readme_cites_the_pinned_results_file(readme):
    assert PINNED.name in readme, "the README must name the results file it quotes"


def test_readme_cites_the_commit_the_results_came_from(readme, results):
    commit = results["run"]["commit"]
    assert commit in readme, f"README should cite commit {commit}"


def test_pinned_results_were_taken_from_a_clean_tree(results):
    assert results["run"]["dirty"] is False, (
        "the pinned results were measured with uncommitted changes, so the "
        "commit they name does not describe the code that ran")


def test_pinned_results_carry_the_timing_protocol(results):
    proto = results["protocol"]
    assert "mx.eval" in proto["timing"]
    assert "no cooldown" in proto["timing"] or "thermal" in proto["timing"]
    assert proto["level0_warmup_reps"] >= 1
    assert "footprint" in proto["copy_baseline"]


# -- the level 0 table -----------------------------------------------------


def test_level0_table_matches_the_measurements(readme, results):
    by_L = {row["L"]: row for row in results["level0"]}
    checked = 0
    for cells in tables(readme):
        if len(cells) != 7 or not cells[1].endswith(("MiB", "KiB")):
            continue
        L = int(num(cells[0]))
        row = by_L[L]
        agrees(cells[2], row["zeta_s"] * 1e3, f"L={L} zeta")
        agrees(cells[3], row["copy_s"] * 1e3, f"L={L} copy")
        agrees(cells[4], row["zeta_over_copy"], f"L={L} passes")
        if row["cpu_s"] is not None:
            agrees(cells[5], row["cpu_s"] * 1e3, f"L={L} cpu")
            agrees(cells[6], row["cpu_s"] / row["zeta_s"], f"L={L} speedup")
        checked += 1
    assert checked == 3, f"expected 3 level-0 rows in the README, found {checked}"


def test_copy_bandwidth_range_claim(readme, results):
    """The README must not quote a single denominator; it quotes the spread."""
    gbps = [r["copy_gbps"] for r in results["level0"]]
    lo, hi = min(gbps), max(gbps)
    assert f"{lo:.1f} GB/s" in readme, f"stated low end should be {lo:.1f} GB/s"
    assert f"{hi:.0f} GB/s" in readme, f"stated high end should be {hi:.0f} GB/s"
    big = [r["copy_gbps"] for r in results["level0"] if r["L"] >= 22]
    assert f"{min(big):.0f}–{max(big):.0f} GB/s" in readme


def test_passes_range_claim(readme, results):
    ratios = [r["zeta_over_copy"] for r in results["level0"] if r["L"] >= 18]
    assert f"{min(ratios):.1f}–{max(ratios):.1f} device copies" in readme


# -- the crossover table ---------------------------------------------------


def test_crossover_table_matches_and_no_row_is_projected(readme, results):
    by_L = {row["L"]: row for row in results["crossover"]}
    checked = 0
    for cells in tables(readme):
        if len(cells) != 4 or not cells[1].endswith("ms") or not cells[3].endswith("×"):
            continue
        L = int(num(cells[0]))
        if L not in by_L:
            continue
        row = by_L[L]
        assert row["dense_ran_to_completion"], (
            f"L={L} is published as a measurement but the dense path did not "
            "run to completion")
        assert row["same_set"] == "yes", f"L={L} paths disagreed"
        agrees(cells[1], row["dense_s"] * 1e3, f"L={L} dense")
        agrees(cells[2], row["struct_s"] * 1e3, f"L={L} structural")
        agrees(cells[3], row["ratio"], f"L={L} ratio")
        checked += 1
    assert checked == 3, f"expected 3 crossover rows, found {checked}"


# -- counts ----------------------------------------------------------------


def test_preset_count_claim(readme, results):
    n = len(results["zeta"])
    assert f"{n} classical presets" in readme
    from apps.zeta.__main__ import PRESETS
    assert len(PRESETS) == n


def test_quickstart_test_count_is_the_real_count(readme):
    """The advertised test count must be what pytest actually collects."""
    import subprocess
    import sys

    m = re.search(r"pytest\s+#\s*(\d+) tests", readme)
    assert m, "the quickstart should advertise a test count"
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT)
    got = re.search(r"(\d+) tests? collected", out.stdout)
    assert got, out.stdout[-2000:]
    assert int(m.group(1)) == int(got.group(1)), (
        f"README says {m.group(1)} tests, pytest collects {got.group(1)}")


# -- the rank claim --------------------------------------------------------


def test_rank_equals_M_claim_matches_the_measured_profile(readme, results):
    """README states rank = M exactly. The grokking instance has M = 1."""
    assert "rank_{F_p} J  =  M" in readme
    for entry in results["grokking_rank_profile"]:
        assert entry["rank"] == 1, "the grokking instance has one output"
        assert all(c.startswith("C'") for c in entry["live_columns"])


# -- the zeta per-preset table ---------------------------------------------


def test_zeta_readme_depths_match_the_measurements(zeta_readme, results):
    by_preset = {row["preset"]: row for row in results["zeta"]}
    checked = 0
    for cells in tables(zeta_readme):
        name = cells[0].strip("`")
        if name not in by_preset:
            continue
        row = by_preset[name]
        if len(cells) < 5:
            continue
        assert int(num(cells[1])) == row["L"], f"{name} L"
        assert int(num(cells[2])) == row["terms_computed"], f"{name} terms"
        assert int(num(cells[3])) == row["exact_upto"], f"{name} exact prefix"
        assert int(num(cells[4])) == row["brute_force_depth"], f"{name} brute depth"
        checked += 1
    assert checked == len(by_preset), (
        f"every preset needs a row with its verified depth; "
        f"{checked} of {len(by_preset)} found")


def test_every_preset_was_actually_verified(results):
    for row in results["zeta"]:
        assert row["matches_brute_force"], row["preset"]
        assert row["brute_force_depth"] >= 4, (
            f"{row['preset']} was only checked to e={row['brute_force_depth']}")
        assert row["matches_closed_form"] in ("OK", "-"), row["preset"]
