"""
Capture real DARWIN frontend footage for the 120s demo.

This script records one clip per scene using Playwright's built-in context video
capture. Clips are recorded from actual dashboard routes and interactions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_playwright():
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "Playwright is required for capture. Install with:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        ) from exc
    return async_playwright


@dataclass(frozen=True)
class SceneCapture:
    scene_id: str
    route: str
    duration_s: int
    side_text: str
    actions: tuple[dict[str, Any], ...]


def parse_scenes(shotlist_path: Path) -> list[SceneCapture]:
    payload = json.loads(shotlist_path.read_text(encoding="utf-8"))
    scenes: list[SceneCapture] = []
    for item in payload["scenes"]:
        scene_id = item["scene_id"]
        route = item["route"]
        duration_s = int(item["duration_s"])
        side_text = item["side_text"]
        actions = scene_actions(scene_id)
        scenes.append(
            SceneCapture(
                scene_id=scene_id,
                route=route,
                duration_s=duration_s,
                side_text=side_text,
                actions=actions,
            )
        )
    return scenes


def scene_actions(scene_id: str) -> tuple[dict[str, Any], ...]:
    # Text-based selectors keep this robust across incremental style changes.
    mapping: dict[str, tuple[dict[str, Any], ...]] = {
        "A": ({"kind": "wait", "seconds": 2.0},),
        "B": (
            {"kind": "click_text", "text": "Play"},
            {"kind": "wait", "seconds": 10.0},
            {"kind": "click_text", "text": "Play"},
            {"kind": "wait", "seconds": 2.0},
        ),
        "C": (
            {"kind": "click_text", "text": "DMs"},
            {"kind": "wait", "seconds": 4.0},
            {"kind": "click_text", "text": "Broadcasts"},
            {"kind": "wait", "seconds": 4.0},
        ),
        "D": (
            {"kind": "click_text", "text": "Family"},
            {"kind": "wait", "seconds": 5.0},
            {"kind": "click_text", "text": "DMs"},
            {"kind": "wait", "seconds": 5.0},
        ),
        "E": (
            {"kind": "click_label", "text": "Optimal moves"},
            {"kind": "wait", "seconds": 8.0},
            {"kind": "click_label", "text": "Optimal moves"},
            {"kind": "wait", "seconds": 2.0},
            {"kind": "click_label", "text": "Optimal moves"},
        ),
        "F": (
            {"kind": "click_text", "text": "Play"},
            {"kind": "wait", "seconds": 11.0},
            {"kind": "click_text", "text": "Play"},
        ),
        "G": (
            {"kind": "click_text", "text": "Highlights only"},
            {"kind": "wait", "seconds": 4.0},
            {"kind": "click_text", "text": "All"},
            {"kind": "wait", "seconds": 4.0},
        ),
        "H": (
            {"kind": "click_text", "text": "Deception"},
            {"kind": "wait", "seconds": 5.0},
            {"kind": "click_text", "text": "Provider"},
            {"kind": "wait", "seconds": 2.0},
        ),
    }
    return mapping.get(scene_id, tuple())


async def maybe_click_text(page, text: str) -> bool:
    # Try button/tab/link first, then free text locator.
    for getter in (
        lambda: page.get_by_role("button", name=text),
        lambda: page.get_by_role("tab", name=text),
        lambda: page.get_by_role("link", name=text),
        lambda: page.get_by_text(text, exact=False),
    ):
        loc = getter().first
        try:
            if await loc.count() > 0:
                await loc.click(timeout=1200)
                return True
        except Exception:
            continue
    return False


async def run_action(page, action: dict[str, Any]) -> None:
    kind = action["kind"]
    if kind == "wait":
        await page.wait_for_timeout(int(action["seconds"] * 1000))
        return
    if kind == "click_text":
        await maybe_click_text(page, action["text"])
        await page.wait_for_timeout(600)
        return
    if kind == "click_label":
        await maybe_click_text(page, action["text"])
        await page.wait_for_timeout(600)
        return
    raise ValueError(f"Unsupported action kind: {kind}")


async def capture_scene(
    playwright,
    base_url: str,
    game_id: str,
    scene: SceneCapture,
    raw_dir: Path,
) -> Path:
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        record_video_dir=str(raw_dir),
        record_video_size={"width": 1920, "height": 1080},
        color_scheme="light",
    )
    page = await context.new_page()

    route = scene.route
    if route in ("/replay", "/analysis", "/investigation"):
        url = f"{base_url}{route}?gameId={game_id}"
    else:
        url = f"{base_url}{route}"
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_timeout(1200)

    elapsed = 0.0
    for action in scene.actions:
        await run_action(page, action)
        if action["kind"] == "wait":
            elapsed += float(action["seconds"])
        else:
            elapsed += 0.6
        if elapsed >= scene.duration_s:
            break

    remaining = max(0.0, scene.duration_s - elapsed)
    if remaining:
        await page.wait_for_timeout(int(remaining * 1000))

    video = page.video
    await context.close()
    await browser.close()
    if video is None:
        raise RuntimeError(f"No video produced for scene {scene.scene_id}")
    src_path = Path(await video.path())
    dst_path = raw_dir / f"scene_{scene.scene_id}.webm"
    if dst_path.exists():
        dst_path.unlink()
    src_path.replace(dst_path)
    return dst_path


async def main_async(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    shotlist = root / "shotlist.json"
    if not shotlist.exists():
        raise FileNotFoundError(
            f"Missing shotlist: {shotlist}. Run python -m scripts.demo_video first."
        )
    scenes = parse_scenes(shotlist)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    async_playwright = load_playwright()
    async with async_playwright() as playwright:
        for scene in scenes:
            print(f"[capture] Scene {scene.scene_id} ({scene.duration_s}s) from {scene.route}")
            out = await capture_scene(playwright, args.base_url, args.game_id, scene, raw_dir)
            print(f"[capture]   wrote {out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture DARWIN demo footage from real UI")
    parser.add_argument("--game-id", required=True, help="Replay game ID")
    parser.add_argument("--base-url", default="http://localhost:3000", help="Dashboard URL")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("media/videos/demo_video"),
        help="Output root used by demo pipeline",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

