-- FPL AI Advisor — database schema (phase 1)
-- Run once via: python -m src.db.init_db

CREATE TABLE IF NOT EXISTS teams (
    id                     INTEGER PRIMARY KEY,
    name                   TEXT NOT NULL,
    short_name             TEXT NOT NULL,
    strength               INTEGER,
    strength_overall_home  INTEGER,
    strength_overall_away  INTEGER,
    strength_attack_home   INTEGER,
    strength_attack_away   INTEGER,
    strength_defence_home  INTEGER,
    strength_defence_away  INTEGER
);

CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY,   -- element_type in the API
    singular_name       TEXT NOT NULL,
    singular_name_short TEXT NOT NULL,
    squad_select        INTEGER,
    squad_min_play      INTEGER,
    squad_max_play      INTEGER
);

CREATE TABLE IF NOT EXISTS gameweeks (
    id                  INTEGER PRIMARY KEY,   -- "event" in the API
    name                TEXT NOT NULL,
    deadline_time       TIMESTAMPTZ,
    is_current          BOOLEAN,
    is_next             BOOLEAN,
    finished            BOOLEAN,
    average_entry_score INTEGER,
    highest_score       INTEGER
);

CREATE TABLE IF NOT EXISTS players (
    id                           INTEGER PRIMARY KEY,
    first_name                   TEXT,
    second_name                  TEXT,
    web_name                     TEXT,
    team_id                      INTEGER REFERENCES teams(id),
    element_type                 INTEGER REFERENCES positions(id),
    now_cost                     INTEGER,   -- price in tenths of a million (55 = £5.5m)
    total_points                 INTEGER,
    form                         REAL,
    points_per_game              REAL,
    selected_by_percent          REAL,
    minutes                      INTEGER,
    goals_scored                 INTEGER,
    assists                      INTEGER,
    clean_sheets                 INTEGER,
    expected_goals               REAL,
    expected_assists             REAL,
    status                       TEXT,      -- a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    chance_of_playing_next_round INTEGER,
    news                         TEXT,
    updated_at                   TIMESTAMPTZ DEFAULT now()
);

-- Time-series snapshots so price/form/ownership history accumulates from day 1.
CREATE TABLE IF NOT EXISTS player_snapshots (
    id                           BIGSERIAL PRIMARY KEY,
    player_id                    INTEGER REFERENCES players(id),
    captured_at                  TIMESTAMPTZ DEFAULT now(),
    gameweek                     INTEGER,
    now_cost                     INTEGER,
    form                         REAL,
    total_points                 INTEGER,
    points_per_game              REAL,
    selected_by_percent          REAL,
    minutes                      INTEGER,
    status                       TEXT,
    chance_of_playing_next_round INTEGER
);
CREATE INDEX IF NOT EXISTS idx_player_snapshots_player
    ON player_snapshots (player_id, captured_at);

-- Historical per-gameweek player performance, loaded from the vaastav dataset.
-- Used to train and backtest the points model. One row per player per fixture
-- (the fixture id is part of the key so double gameweeks keep both rows).
CREATE TABLE IF NOT EXISTS player_gameweek_history (
    id                         BIGSERIAL PRIMARY KEY,
    season                     TEXT NOT NULL,
    gw                         INTEGER,
    element                    INTEGER NOT NULL,   -- player id within that season
    fixture                    INTEGER,
    name                       TEXT,
    position                   TEXT,
    team                       TEXT,
    opponent_team              INTEGER,
    was_home                   BOOLEAN,
    minutes                    INTEGER,
    total_points               INTEGER,
    goals_scored               INTEGER,
    assists                    INTEGER,
    clean_sheets               INTEGER,
    goals_conceded             INTEGER,
    bonus                      INTEGER,
    bps                        INTEGER,
    saves                      INTEGER,
    starts                     INTEGER,
    xp                         REAL,               -- FPL's own expected points
    expected_goals             REAL,
    expected_assists           REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded    REAL,
    ict_index                  REAL,
    influence                  REAL,
    creativity                 REAL,
    threat                     REAL,
    value                      INTEGER,            -- price that GW (tenths of a million)
    selected                   INTEGER,
    transfers_in               INTEGER,
    transfers_out              INTEGER,
    kickoff_time               TIMESTAMPTZ,
    UNIQUE (season, element, fixture)
);
CREATE INDEX IF NOT EXISTS idx_pgh_season_gw ON player_gameweek_history (season, gw);
CREATE INDEX IF NOT EXISTS idx_pgh_element  ON player_gameweek_history (season, element);

CREATE TABLE IF NOT EXISTS fixtures (
    id                INTEGER PRIMARY KEY,
    event             INTEGER,             -- gameweek (NULL until scheduled)
    kickoff_time      TIMESTAMPTZ,
    team_h            INTEGER REFERENCES teams(id),
    team_a            INTEGER REFERENCES teams(id),
    team_h_difficulty INTEGER,             -- FDR for the home side
    team_a_difficulty INTEGER,             -- FDR for the away side
    team_h_score      INTEGER,
    team_a_score      INTEGER,
    finished          BOOLEAN,
    updated_at        TIMESTAMPTZ DEFAULT now()
);
