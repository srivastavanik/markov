#!/usr/bin/env python3
"""
Run N sequential Darwin games via the API to build up trace data for fine-tuning.

Requires the API server to be running (python -m scripts.run_api).

Usage:
  python -m scripts.batch_games             # 10 games (default)
  python -m scripts.batch_games --count 20  # 20 games
  python -m scripts.batch_games --api http://localhost:8000  # custom API URL

Each full 12-agent game produces ~20-30 usable reasoning traces.
Games run sequentially (one at a time) to avoid overloading LLM APIs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

DEFAULT_API = os.getenv("DARWIN_API_URL", "http://localhost:8000")


def _check_api(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/games", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _get_games(base_url: str) -> list[dict]:
    """Get games list, handling both list and dict response formats."""
    r = requests.get(f"{base_url}/api/games", timeout=5)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    return data.get("games", [])


def _start_game(base_url: str) -> str:
    """Start a game, return the game_id."""
    r = requests.post(
        f"{base_url}/api/games",
        json={"mode": "full", "verbose": False},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return data["game_id"]


def _poll_game(base_url: str, game_id: str, timeout_s: int = 600) -> str:
    """Poll until game completes. Returns final status."""
    start = time.monotonic()
    last_status = "unknown"
    while time.monotonic() - start < timeout_s:
        try:
            games = _get_games(base_url)
            match = next((g for g in games if g["game_id"] == game_id), None)
            if match:
                last_status = match["status"]
                if last_status in ("completed", "failed"):
                    return last_status
        except Exception as e:
            print(f"    Poll error: {e}")
        time.sleep(5)
    return last_status


def _count_traces(base_url: str) -> int:
    """Quick count of usable traces in Supabase."""
    try:
        from supabase import create_client
        url = f"https://{os.getenv('SUPABASE_PROJECT_REF', 'yyistnxvozjmqmawdent')}.supabase.co"
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not key:
            return -1
        sb = create_client(url, key)
        all_rt = (
            sb.table("game_rounds")
            .select("reasoning_traces_json")
            .not_.is_("reasoning_traces_json", "null")
            .execute()
        )
        count = 0
        for row in all_rt.data or []:
            rt_raw = row.get("reasoning_traces_json")
            if isinstance(rt_raw, str):
                try:
                    rt = json.loads(rt_raw)
                except json.JSONDecodeError:
                    continue
            else:
                rt = rt_raw
            if not isinstance(rt, dict):
                continue
            for td in rt.values():
                if not isinstance(td, dict):
                    continue
                text = td.get("thinking_trace") or td.get("reasoning_summary") or ""
                if len(text.strip()) >= 50:
                    count += 1
        return count
    except Exception as e:
        print(f"  (Trace count unavailable: {e})")
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch Darwin games to build trace dataset")
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of games to run (default: 10)")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds per game (default: 600)")
    args = parser.parse_args()

    if not _check_api(args.api):
        print(f"ERROR: API not reachable at {args.api}", file=sys.stderr)
        print("  Start it with: python -m scripts.run_api", file=sys.stderr)
        sys.exit(1)

    initial_traces = _count_traces(args.api)
    if initial_traces >= 0:
        print(f"Current trace count: {initial_traces}")

    print(f"\nRunning {args.count} sequential games via {args.api}")
    print(f"Each game takes ~2-5 min depending on model latency.\n")

    completed = 0
    failed = 0

    for i in range(1, args.count + 1):
        print(f"[{i}/{args.count}] Starting game...", end=" ", flush=True)
        try:
            game_id = _start_game(args.api)
            print(f"{game_id}", end=" ", flush=True)
        except Exception as e:
            print(f"FAILED to start: {e}")
            failed += 1
            continue

        status = _poll_game(args.api, game_id, timeout_s=args.timeout)
        elapsed = ""

        if status == "completed":
            completed += 1
            print(f"-- completed")
        elif status == "failed":
            failed += 1
            print(f"-- FAILED")
        else:
            failed += 1
            print(f"-- timed out (status: {status})")

        # Brief pause between games to let things settle
        if i < args.count:
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Batch complete: {completed} completed, {failed} failed")

    final_traces = _count_traces(args.api)
    if final_traces >= 0:
        new_traces = final_traces - initial_traces if initial_traces >= 0 else 0
        print(f"Total traces: {final_traces} (+{new_traces} new)")
        if final_traces >= 500:
            print(f"\nReady for fine-tuning. Run:")
        elif final_traces >= 200:
            print(f"\nMinimum viable for fine-tuning. Run:")
        else:
            print(f"\nStill building up. Run more games or proceed with what you have:")
        print(f"  python -m scripts.finetune_classifier pull")
        print(f"  python -m scripts.finetune_classifier submit-batch")


if __name__ == "__main__":
    main()
