"""Checks for the bottom panel of Figure 1.

The published panel had its four series typed in as literals. These tests hold
the recomputed values to those literals, so the panel can never drift from what
was printed without a test failing.
"""

from __future__ import annotations

import pytest

from superarc.complexity_measures import (
    PUBLISHED_TIERS,
    average_length_of_strings,
    measures,
    sequences_definitons,
)

# Exactly as typed into the published plotting cell.
PUBLISHED = {
    "BDM": [471.544, 494.951, 549.678],
    "Shannon": [3.65, 3.67, 3.91],
    "zip": [46.7, 49, 59.63],
    "lzw": [118.1, 121.6, 131.15],
}


@pytest.fixture(scope="module")
def computed():
    return measures()


@pytest.mark.parametrize("name", ["Shannon", "zip", "lzw"])
def test_reproduces_the_published_values(computed, name):
    for got, printed in zip(computed[name], PUBLISHED[name]):
        # Tolerance is one unit in the last digit printed, not half a unit: the
        # published numbers were shortened inconsistently. 3.9134 was rounded to
        # 3.91, but 59.63636 was truncated to 59.63 rather than rounded to 59.64.
        assert got == pytest.approx(printed, abs=10 ** -_decimals(printed))


def test_bdm_reproduces_the_published_values_except_one_transposition(computed):
    """The third BDM point was printed as 549.678; it computes to 549.687.

    Two digits transposed while copying the number into the plotting cell. Pinned
    here so the discrepancy stays visible and cannot be mistaken for drift.
    """
    assert computed["BDM"][0] == pytest.approx(471.544, abs=0.001)
    assert computed["BDM"][1] == pytest.approx(494.951, abs=0.001)
    assert computed["BDM"][2] == pytest.approx(549.687, abs=0.001)
    assert computed["BDM"][2] != pytest.approx(549.678, abs=0.001)


def test_every_measure_agrees_that_complexity_increases(computed):
    """The claim the panel exists to support.

    Four independent measures -- one algorithmic, one statistical, two ordinary
    compressors -- must all rank the tiers the same way, or the tiers are not
    ordered by complexity at all.
    """
    for name, series in computed.items():
        assert series == sorted(series), f"{name} is not monotone: {series}"


def test_published_tiers_skip_the_third_pool():
    """Tiers 3 and 4 are swapped and the new tier 4 dropped -- so 1, 2, 4."""
    assert PUBLISHED_TIERS == (0, 1, 3)
    assert len(sequences_definitons()) == 4


def test_average_length_of_empty_list_is_zero():
    assert average_length_of_strings([]) == 0


def _decimals(value: float) -> int:
    text = repr(float(value))
    return len(text.split(".")[1].rstrip("0"))
