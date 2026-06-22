"""Unit tests for the advisor's decision helpers.

All pure functions: no database, model or network. We feed synthetic FPL
history dicts / chip-timing tables and assert the logic.
"""
import pandas as pd

from src.advisor.personal import (
    estimate_free_transfers, chips_available, chips_used,
    recommend_chip, optimize_xi,
)
from src.config import MAX_FT, CHIP_LABEL, HALF_DEADLINE, SQUAD, XI_MIN, XI_MAX


def _history(transfers=None, chips=None) -> dict:
    """Build a minimal FPL history dict.  transfers: {gw: count}."""
    transfers = transfers or {}
    return {
        "current": [{"event": g, "event_transfers": t} for g, t in transfers.items()],
        "chips": chips or [],
    }


# ----------------------------- free transfers -----------------------------
def test_ft_is_one_in_gw1():
    assert estimate_free_transfers(_history(), 1) == 1


def test_ft_accumulates_below_cap():
    # no transfers made: 1 at the start, +1 per prior week -> 3 entering GW3.
    assert estimate_free_transfers(_history(), 3) == 3


def test_ft_never_exceeds_cap():
    # many idle weeks would accumulate past the cap, but it must hold at MAX_FT.
    assert estimate_free_transfers(_history(), 30) == MAX_FT
    assert estimate_free_transfers(_history(), 30) <= MAX_FT


def test_ft_afcon_gw16_is_five_even_after_heavy_use():
    # spending every week keeps the count at ~1, but GW16 hard-resets to 5.
    heavy = _history(transfers={g: 2 for g in range(1, 16)})
    assert estimate_free_transfers(heavy, 16) == 5


def test_ft_heavy_use_keeps_count_low():
    heavy = _history(transfers={g: 2 for g in range(1, 10)})
    assert estimate_free_transfers(heavy, 10) == 1


def test_wildcard_does_not_inflate_ft():
    # a wildcard resets accumulation; entering the next week stays small.
    h = _history(transfers={1: 3, 2: 3}, chips=[{"event": 3, "name": "wildcard"}])
    assert estimate_free_transfers(h, 4) <= 2


# --------------------------------- chips ----------------------------------
def test_all_chips_available_with_none_used():
    available, deadline, half = chips_available(_history(), 5)
    assert set(available) == set(CHIP_LABEL)
    assert deadline == HALF_DEADLINE[1]
    assert half == 1


def test_used_chip_is_excluded_and_reported():
    h = _history(chips=[{"event": 3, "name": "wildcard"}])
    available, _, _ = chips_available(h, 5)
    assert "wildcard" not in available
    assert chips_used(h, 5) == {"wildcard": 3}


def test_second_half_resets_chip_availability():
    # a wildcard used in the first half does not count against the second half.
    h = _history(chips=[{"event": 3, "name": "wildcard"}])
    available, deadline, half = chips_available(h, 25)
    assert set(available) == set(CHIP_LABEL)
    assert deadline == HALF_DEADLINE[2]
    assert half == 2


def test_future_chip_not_counted_as_used_yet():
    # a chip the API shows at GW10 is not "used" when advising for GW8.
    h = _history(chips=[{"event": 10, "name": "bboost"}])
    available, _, _ = chips_available(h, 8)
    assert "bboost" in available
    assert chips_used(h, 8) == {}


# ----------------------------- chip timing --------------------------------
def test_recommend_chip_when_it_peaks_now():
    timing = {"bboost": [(5, 10.0), (6, 9.0)]}      # best week is GW5
    assert recommend_chip(timing, 5) == "bboost"


def test_no_recommendation_when_peak_is_later():
    timing = {"freehit": [(8, 12.0), (5, 7.0)]}     # best week is GW8
    assert recommend_chip(timing, 5) is None


def test_highest_value_wins_when_several_peak_now():
    timing = {"bboost": [(5, 10.0)], "3xc": [(5, 8.0)]}
    assert recommend_chip(timing, 5) == "bboost"


def test_recommend_chip_handles_empty_inputs():
    assert recommend_chip({}, 5) is None
    assert recommend_chip({"bboost": []}, 5) is None


# --------------------------- starting XI from a squad ---------------------
def _squad() -> pd.DataFrame:
    rows = []
    for pos, n in SQUAD.items():
        for j in range(n):
            rows.append({"name": f"{pos}{j}", "position": pos, "pred": float(j + 1)})
    return pd.DataFrame(rows)


def test_optimize_xi_picks_valid_eleven_with_one_captain():
    value, xi = optimize_xi(_squad())
    starters = xi.query("starting == 1")
    assert len(starters) == 11
    assert xi["captain"].sum() == 1
    by_pos = starters["position"].value_counts().to_dict()
    assert by_pos.get("GK", 0) == 1
    for pos in ("DEF", "MID", "FWD"):
        assert XI_MIN[pos] <= by_pos.get(pos, 0) <= XI_MAX[pos]
    # captain is the highest-projected starter
    cap = xi.query("captain == 1").iloc[0]
    assert cap["pred"] == starters["pred"].max()
    assert value > 0
