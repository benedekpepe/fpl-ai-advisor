"""Feature engineering for the expected-points model.

Turns the raw per-gameweek history into a feature table where every feature
uses only information available *before* the target gameweek (no leakage).

The golden rule here: for any rolling/cumulative stat we shift by one game
within each (season, player) group, so the current game's own outcome never
leaks into its own features.
"""
import pandas as pd

# Base stats we summarise as recent form (rolling means over prior games).
ROLL_BASES = [
    "total_points", "minutes", "expected_goals", "expected_assists",
    "expected_goal_involvements", "bps", "ict_index",
]
ROLL_WINDOWS = [3, 5]

POSITIONS = ["GK", "DEF", "MID", "FWD"]


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with engineered features + the target (total_points)."""
    df = history.copy()

    # Stable chronological order within each player-season.
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    df = df.sort_values(["season", "element", "kickoff_time", "gw", "fixture"])

    grp = df.groupby(["season", "element"], sort=False)

    # Rolling form features (shift(1) => strictly prior games only).
    for base in ROLL_BASES:
        for w in ROLL_WINDOWS:
            df[f"roll_{base}_{w}"] = grp[base].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # Share of recent games started (nailed-on proxy).
    df["starts_rate_5"] = grp["starts"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # Season-to-date signals.
    df["games_played"] = grp.cumcount()  # number of prior games this season
    df["cum_ppg"] = grp["total_points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Fixture context known before kickoff.
    df["is_home"] = df["was_home"].astype("float")
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype("float")

    # Need at least one prior game for the form features to mean anything.
    df = df[df["games_played"] >= 1].copy()
    return df


def feature_columns() -> list[str]:
    cols = []
    for base in ROLL_BASES:
        for w in ROLL_WINDOWS:
            cols.append(f"roll_{base}_{w}")
    cols += ["starts_rate_5", "games_played", "cum_ppg", "is_home", "value"]
    cols += [f"pos_{p}" for p in POSITIONS]
    return cols


TARGET = "total_points"
