"""
DARWIN demo video orchestrator (from-scratch scene architecture).

This script does not render synthetic scenes. It defines the canonical 120-second
shot map and orchestrates two stages:
1) Real frontend capture (`scripts/capture_demo_video.py`)
2) Cinematic assembly (`scripts/assemble_demo_video.py`)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DemoScene:
    scene_id: str
    title: str
    start_s: int
    duration_s: int
    route: str
    primary_focus: str
    side_text: str
    required_features: tuple[str, ...]


SCENES: tuple[DemoScene, ...] = (
    DemoScene(
        scene_id="A",
        title="System Establishment",
        start_s=0,
        duration_s=12,
        route="/replay",
        primary_focus="Top bar + live board shell",
        side_text="DARWIN runtime online",
        required_features=("runtime_status", "board_shell"),
    ),
    DemoScene(
        scene_id="B",
        title="Board Dynamics + Piece Movement",
        start_s=12,
        duration_s=18,
        route="/replay",
        primary_focus="Board transitions and movement pressure",
        side_text="Movement pressure builds",
        required_features=("piece_movement",),
    ),
    DemoScene(
        scene_id="C",
        title="Broadcast + DMs In Parallel",
        start_s=30,
        duration_s=18,
        route="/live",
        primary_focus="Broadcast stream + DM thread list",
        side_text="Public vs private channels",
        required_features=("broadcast", "dms"),
    ),
    DemoScene(
        scene_id="D",
        title="Multi-Turn Intra-Round Conversations",
        start_s=48,
        duration_s=18,
        route="/live",
        primary_focus="Family and DM reply cadence in same round",
        side_text="Multi-turn same-round exchange",
        required_features=("multi_turn_conversations",),
    ),
    DemoScene(
        scene_id="E",
        title="Optimal Moves / Decision Matrix",
        start_s=66,
        duration_s=16,
        route="/live",
        primary_focus="Optimal moves overlays + recommendations",
        side_text="Decision matrix in context",
        required_features=("optimal_moves", "decision_matrix"),
    ),
    DemoScene(
        scene_id="F",
        title="Resolve + Bottom Kill Cam",
        start_s=82,
        duration_s=16,
        route="/replay",
        primary_focus="Elimination outcomes + kill timeline",
        side_text="Resolution and kill cam",
        required_features=("kill_cam", "eliminations"),
    ),
    DemoScene(
        scene_id="G",
        title="Sentry Flagging CoT",
        start_s=98,
        duration_s=14,
        route="/investigation",
        primary_focus="Thought stream badges + highlight severity",
        side_text="Sentry flags reasoning",
        required_features=("sentry_flags", "cot_badges"),
    ),
    DemoScene(
        scene_id="H",
        title="Deception Delta + Closing",
        start_s=112,
        duration_s=8,
        route="/analysis",
        primary_focus="Deception chart and concise close",
        side_text="Deception delta summary",
        required_features=("deception_delta",),
    ),
)

EXPECTED_TOTAL_SECONDS = 120


def ensure_scene_timing(scenes: Sequence[DemoScene]) -> None:
    if not scenes:
        raise ValueError("No scenes configured.")
    for idx, scene in enumerate(scenes):
        if scene.duration_s <= 0:
            raise ValueError(f"Scene {scene.scene_id} has non-positive duration.")
        if idx == 0 and scene.start_s != 0:
            raise ValueError("First scene must start at t=0.")
        if idx > 0:
            prev = scenes[idx - 1]
            if scene.start_s != prev.start_s + prev.duration_s:
                raise ValueError(
                    f"Scene timing gap/overlap: {prev.scene_id} -> {scene.scene_id}."
                )
    total = scenes[-1].start_s + scenes[-1].duration_s
    if total != EXPECTED_TOTAL_SECONDS:
        raise ValueError(f"Expected 120s total, got {total}s.")


def output_root(custom: Path | None = None) -> Path:
    return custom or Path("media/videos/demo_video")


def write_shotlist(root: Path, scenes: Sequence[DemoScene]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shotlist_path = root / "shotlist.json"
    payload = {
        "project": "DARWIN cinematic demo",
        "total_seconds": EXPECTED_TOTAL_SECONDS,
        "scenes": [asdict(scene) for scene in scenes],
    }
    shotlist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return shotlist_path


def write_timeline_markdown(root: Path, scenes: Sequence[DemoScene]) -> Path:
    timeline_path = root / "timeline.md"
    lines = [
        "# DARWIN Demo Timeline",
        "",
        "| Scene | Timecode | Duration | Focus | Route |",
        "|---|---:|---:|---|---|",
    ]
    for scene in scenes:
        start = scene.start_s
        end = scene.start_s + scene.duration_s
        lines.append(
            f"| {scene.scene_id} | {start:03d}-{end:03d}s | {scene.duration_s}s | "
            f"{scene.primary_focus} | `{scene.route}` |"
        )
    lines += ["", f"Total: **{EXPECTED_TOTAL_SECONDS}s**"]
    timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return timeline_path


def run_subprocess(cmd: list[str], dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="DARWIN demo video pipeline orchestrator")
    parser.add_argument("--game-id", default="game_7a22eaea9dd7", help="Replay game ID")
    parser.add_argument("--base-url", default="http://localhost:3000", help="Dashboard URL")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("media/videos/demo_video"),
        help="Pipeline output root directory",
    )
    parser.add_argument("--capture", action="store_true", help="Run capture stage")
    parser.add_argument("--assemble", action="store_true", help="Run assembly stage")
    parser.add_argument("--all", action="store_true", help="Run both capture and assembly")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    ensure_scene_timing(SCENES)
    root = output_root(args.output_root)
    shotlist_path = write_shotlist(root, SCENES)
    timeline_path = write_timeline_markdown(root, SCENES)
    print(f"Wrote shotlist: {shotlist_path}")
    print(f"Wrote timeline: {timeline_path}")

    run_capture = args.capture or args.all
    run_assemble = args.assemble or args.all

    if run_capture:
        run_subprocess(
            [
                sys.executable,
                "-m",
                "scripts.capture_demo_video",
                "--game-id",
                args.game_id,
                "--base-url",
                args.base_url,
                "--output-root",
                str(root),
            ],
            dry_run=args.dry_run,
        )

    if run_assemble:
        run_subprocess(
            [
                sys.executable,
                "-m",
                "scripts.assemble_demo_video",
                "--output-root",
                str(root),
            ],
            dry_run=args.dry_run,
        )

    if not (run_capture or run_assemble):
        print("No stage selected. Use --capture, --assemble, or --all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

