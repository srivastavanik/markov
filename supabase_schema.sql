-- MARKOV Supabase Schema
-- Run this in the Supabase SQL Editor to create all required tables.

-- ==========================================================================
-- Games
-- ==========================================================================
CREATE TABLE IF NOT EXISTS games (
  id              TEXT PRIMARY KEY,
  series_id       TEXT,
  series_type     TEXT DEFAULT 'standard',
  started_at      TIMESTAMPTZ DEFAULT now(),
  total_rounds    INT,
  winner_id       TEXT,
  winner_name     TEXT,
  winner_provider TEXT,
  config_json     JSONB DEFAULT '{}',
  cost_json       JSONB DEFAULT '{}',
  metrics_json    JSONB DEFAULT '{}',
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- ==========================================================================
-- Game Agents — one row per agent per game
-- ==========================================================================
CREATE TABLE IF NOT EXISTS game_agents (
  id               TEXT PRIMARY KEY,   -- "{game_id}_{agent_id}"
  game_id          TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  agent_id         TEXT NOT NULL,
  agent_name       TEXT NOT NULL,
  family           TEXT NOT NULL,
  provider         TEXT NOT NULL,
  model            TEXT NOT NULL,
  tier             INT NOT NULL DEFAULT 1,
  alive            BOOLEAN DEFAULT TRUE,
  eliminated_round INT,
  eliminated_by    TEXT,
  rounds_survived  INT DEFAULT 0,
  created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_game_agents_game ON game_agents(game_id);

-- ==========================================================================
-- Game Rounds — one row per round per game
-- Includes reasoning traces and model responses
-- ==========================================================================
CREATE TABLE IF NOT EXISTS game_rounds (
  id                      TEXT PRIMARY KEY,   -- "{game_id}_r{round_num}"
  game_id                 TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  round_num               INT NOT NULL,
  thoughts_json           JSONB DEFAULT '{}',
  messages_json           JSONB DEFAULT '[]',
  events_json             JSONB DEFAULT '[]',
  actions_json            JSONB DEFAULT '{}',
  reasoning_traces_json   JSONB DEFAULT '{}',
  family_discussions_json JSONB DEFAULT '[]',
  created_at              TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_game_rounds_game ON game_rounds(game_id);

-- ==========================================================================
-- Game Analysis — per-round agent analysis
-- ==========================================================================
CREATE TABLE IF NOT EXISTS game_analysis (
  id            TEXT PRIMARY KEY,   -- "{game_id}_a{round_num}"
  game_id       TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  round_num     INT NOT NULL,
  analysis_json JSONB DEFAULT '{}',
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_game_analysis_game ON game_analysis(game_id);

-- ==========================================================================
-- Game Highlights — flagged moments
-- ==========================================================================
CREATE TABLE IF NOT EXISTS game_highlights (
  id              TEXT PRIMARY KEY,   -- "{game_id}_h{index}"
  game_id         TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  round_num       INT NOT NULL,
  agent_id        TEXT,
  highlight_type  TEXT,
  severity        TEXT,   -- "critical", "high", "medium"
  description     TEXT,
  excerpt         TEXT,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_game_highlights_game ON game_highlights(game_id);

-- ==========================================================================
-- Series — aggregate metrics across multiple games
-- ==========================================================================
CREATE TABLE IF NOT EXISTS series (
  id                       TEXT PRIMARY KEY,
  series_type              TEXT DEFAULT 'standard',
  num_games                INT DEFAULT 0,
  aggregate_metrics_json   JSONB DEFAULT '{}',
  created_at               TIMESTAMPTZ DEFAULT now()
);
