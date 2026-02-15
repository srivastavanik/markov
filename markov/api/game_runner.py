from __future__ import annotations

import asyncio
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from markov.config import GameConfig, load_game_config
from markov.orchestrator import run_game_llm
from markov.server import GameBroadcaster

logger = logging.getLogger("markov.game_runner")

GameMode = Literal["full", "quick"]
GameStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GAMES_DIR = _PROJECT_ROOT / "data" / "games"
_SERIES_DIR = _PROJECT_ROOT / "data" / "series"


@dataclass
class GameJob:
    game_id: str
    mode: GameMode
    status: GameStatus = "queued"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    output_dir: str | None = None
    error: str | None = None
    cancel_requested: bool = False
    winner: str | None = None
    total_rounds: int | None = None
    task: asyncio.Task | None = None

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "mode": self.mode,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "output_dir": self.output_dir,
            "error": self.error,
            "winner": self.winner,
            "total_rounds": self.total_rounds,
        }


class GameRunner:
    def __init__(self, broadcaster: GameBroadcaster) -> None:
        self.broadcaster = broadcaster
        self.jobs: dict[str, GameJob] = {}
        _GAMES_DIR.mkdir(parents=True, exist_ok=True)
        _SERIES_DIR.mkdir(parents=True, exist_ok=True)

    def start_game(self, mode: GameMode = "full", verbose: bool = False) -> GameJob:
        game_id = f"game_{uuid4().hex[:12]}"
        job = GameJob(game_id=game_id, mode=mode)
        self.jobs[game_id] = job
        task = asyncio.create_task(self._run_job(job, verbose=verbose), name=f"game-{game_id}")
        task.add_done_callback(self._on_task_done)
        job.task = task
        return job

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        """Log unhandled exceptions from background game tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("Background game task %s raised: %s", task.get_name(), exc, exc_info=exc)

    def request_cancel(self, game_id: str) -> GameJob | None:
        job = self.jobs.get(game_id)
        if not job:
            return None
        if job.status in {"completed", "failed", "cancelled"}:
            return job
        job.cancel_requested = True
        return job

    def get_job(self, game_id: str) -> GameJob | None:
        return self.jobs.get(game_id)

    def list_jobs(self) -> list[GameJob]:
        jobs = list(self.jobs.values())
        known_ids = {j.game_id for j in jobs}
        for game_json in sorted(_GAMES_DIR.glob("*/game.json"), reverse=True):
            game_id = game_json.parent.name
            if game_id in known_ids:
                continue
            started = datetime.fromtimestamp(game_json.stat().st_mtime, tz=timezone.utc)
            jobs.append(
                GameJob(
                    game_id=game_id,
                    mode="full",
                    status="completed",
                    started_at=started,
                    ended_at=started,
                    output_dir=str(game_json.parent),
                )
            )
        return sorted(jobs, key=lambda j: j.started_at, reverse=True)

    def get_replay_payload(self, game_id: str) -> dict | None:
        game_dir = _GAMES_DIR / game_id
        json_path = game_dir / "game.json"
        if not json_path.exists():
            return None
        with open(json_path) as f:
            payload = json.load(f)
        analysis_path = game_dir / "analysis.json"
        highlights_path = game_dir / "highlights.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis_data = json.load(f)
            rounds = payload.get("rounds", [])
            if isinstance(analysis_data, list):
                by_round = {entry.get("round"): entry for entry in analysis_data if isinstance(entry, dict)}
                for round_entry in rounds:
                    rnum = round_entry.get("round")
                    if rnum in by_round:
                        round_entry["analysis"] = by_round[rnum].get("analysis", round_entry.get("analysis", {}))
        if highlights_path.exists():
            with open(highlights_path) as f:
                highlights_data = json.load(f)
            rounds = payload.get("rounds", [])
            if isinstance(highlights_data, list):
                by_round = {}
                for h in highlights_data:
                    r = h.get("round")
                    if r is None:
                        continue
                    by_round.setdefault(r, []).append(h)
                for round_entry in rounds:
                    rnum = round_entry.get("round")
                    if rnum in by_round:
                        round_entry["highlights"] = by_round[rnum]
        return payload

    def get_metrics_payload(self, game_id: str) -> dict | None:
        metrics_path = _GAMES_DIR / game_id / "metrics.json"
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            return json.load(f)

    def get_analysis_payload(self, game_id: str) -> dict | list | None:
        analysis_path = _GAMES_DIR / game_id / "analysis.json"
        if not analysis_path.exists():
            return None
        with open(analysis_path) as f:
            return json.load(f)

    def list_series(self) -> list[dict]:
        rows: list[dict] = []
        for directory in sorted(_SERIES_DIR.glob("*"), reverse=True):
            if not directory.is_dir():
                continue
            cfg = directory / "series_config.json"
            agg = directory / "aggregate_metrics.json"
            cfg_data: dict = {}
            agg_data: dict | None = None
            if cfg.exists():
                with open(cfg) as f:
                    cfg_data = json.load(f)
            if agg.exists():
                with open(agg) as f:
                    agg_data = json.load(f)
            rows.append({
                "series_id": directory.name,
                "series_type": cfg_data.get("series_type", "unknown"),
                "provider": cfg_data.get("provider"),
                "num_games": cfg_data.get("num_games", 0),
                "created_at": datetime.fromtimestamp(directory.stat().st_mtime, tz=timezone.utc),
                "aggregate_metrics": agg_data,
            })
        return rows

    def get_series_detail(self, series_id: str) -> dict | None:
        directory = _SERIES_DIR / series_id
        if not directory.exists() or not directory.is_dir():
            return None
        cfg = directory / "series_config.json"
        agg = directory / "aggregate_metrics.json"
        cfg_data: dict = {}
        agg_data: dict | None = None
        if cfg.exists():
            with open(cfg) as f:
                cfg_data = json.load(f)
        if agg.exists():
            with open(agg) as f:
                agg_data = json.load(f)
        games: list[dict] = []
        for game_dir in sorted(directory.glob("game_*")):
            if not game_dir.is_dir():
                continue
            game_json = game_dir / "game.json"
            if not game_json.exists():
                continue
            started = datetime.fromtimestamp(game_json.stat().st_mtime, tz=timezone.utc)
            games.append({
                "game_id": game_dir.name,
                "status": "completed",
                "mode": "full",
                "started_at": started,
                "ended_at": started,
                "output_dir": str(game_dir),
                "error": None,
                "winner": None,
                "total_rounds": None,
            })
        return {
            "series_id": series_id,
            "series_type": cfg_data.get("series_type", "unknown"),
            "provider": cfg_data.get("provider"),
            "num_games": cfg_data.get("num_games", len(games)),
            "config": cfg_data.get("config", {}),
            "games": games,
            "aggregate_metrics": agg_data,
        }

    async def _run_job(self, job: GameJob, verbose: bool) -> None:
        job.status = "running"
        config = self._build_config(job.mode)
        logger.info("[%s] Game starting (mode=%s)", job.game_id, job.mode)
        try:
            state, game_logger = await run_game_llm(
                config=config,
                verbose=verbose,
                broadcaster=self.broadcaster,
                game_id=job.game_id,
                should_stop=lambda: job.cancel_requested,
            )
            game_dir = _GAMES_DIR / job.game_id
            game_logger.save(path=game_dir, agents=state.agents)
            job.output_dir = str(game_dir)
            job.total_rounds = state.round_num
            job.winner = state.winner.name if state.winner else None
            job.status = "cancelled" if job.cancel_requested else "completed"
            logger.info("[%s] Game finished: status=%s winner=%s rounds=%s",
                        job.game_id, job.status, job.winner, job.total_rounds)
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[%s] GAME FAILED: %s: %s\n%s",
                job.game_id,
                type(exc).__name__,
                exc,
                traceback.format_exc(),
            )
        finally:
            job.ended_at = datetime.now(timezone.utc)

    def _build_config(self, mode: GameMode) -> GameConfig:
        config = load_game_config()
        if mode == "quick":
            return GameConfig(
                grid_size=config.grid_size,
                max_rounds=10,
                stalemate_threshold=8,
                discussion_rounds=1,
                families=config.families,
                no_family_discussion=config.no_family_discussion,
                series_type=config.series_type,
            )
        return config

