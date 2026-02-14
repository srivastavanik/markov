from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from markov.config import GameConfig, load_game_config
from markov.orchestrator import run_game_llm
from markov.server import GameBroadcaster

GameMode = Literal["full", "quick"]
GameStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_GAMES_DIR = _PROJECT_ROOT / "data" / "games"


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

    def start_game(self, mode: GameMode = "full", verbose: bool = False) -> GameJob:
        game_id = f"game_{uuid4().hex[:12]}"
        job = GameJob(game_id=game_id, mode=mode)
        self.jobs[game_id] = job
        job.task = asyncio.create_task(self._run_job(job, verbose=verbose))
        return job

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
            return json.load(f)

    async def _run_job(self, job: GameJob, verbose: bool) -> None:
        job.status = "running"
        config = self._build_config(job.mode)
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
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
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

