"""
Assemble captured DARWIN scene clips into a cinematic 1080p60 master.

Inputs:
- media/videos/demo_video/shotlist.json
- media/videos/demo_video/raw/scene_*.webm

Output:
- media/videos/demo_video/1080p60/DarwinDemo.mp4
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


FPS = 60
WIDTH = 1920
HEIGHT = 1080
FONT_PATHS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
)


@dataclass(frozen=True)
class SceneMeta:
    scene_id: str
    duration_s: int
    title: str
    side_text: str


def run(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}).")


def load_scenes(shotlist: Path) -> list[SceneMeta]:
    payload = json.loads(shotlist.read_text(encoding="utf-8"))
    scenes = []
    for row in payload["scenes"]:
        scenes.append(
            SceneMeta(
                scene_id=row["scene_id"],
                duration_s=int(row["duration_s"]),
                title=row["title"],
                side_text=row["side_text"],
            )
        )
    return scenes


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_PATHS:
        p = Path(font_path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def make_overlay_image(scene: SceneMeta, out_path: Path) -> Path:
    w, h = WIDTH, HEIGHT
    image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    panel_x, panel_y, panel_w, panel_h = 1240, 72, 620, 138
    draw.rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        fill=(0, 0, 0, 100),
    )
    draw.rectangle(
        [panel_x, panel_y, panel_x + 4, panel_y + panel_h],
        fill=(255, 255, 255, 140),
    )

    title = f"Scene {scene.scene_id} - {scene.title}"
    subtitle = scene.side_text
    title_font = load_font(44)
    subtitle_font = load_font(32)
    tx, ty = panel_x + 30, panel_y + 24
    sx, sy = panel_x + 30, panel_y + 84

    # Soft glow using blurred white text pass.
    glow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.text((tx, ty), title, font=title_font, fill=(255, 255, 255, 165))
    glow_draw.text((sx, sy), subtitle, font=subtitle_font, fill=(255, 255, 255, 150))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=2.2))
    image.alpha_composite(glow_layer)

    draw = ImageDraw.Draw(image)
    draw.text((tx, ty), title, font=title_font, fill=(255, 255, 255, 245))
    draw.text((sx, sy), subtitle, font=subtitle_font, fill=(255, 255, 255, 230))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def scene_filter(scene: SceneMeta) -> str:
    return ",".join(
        [
            f"fps={FPS}",
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase",
            f"crop={WIDTH}:{HEIGHT}",
            "eq=contrast=1.04:brightness=0.01:saturation=1.03",
            "unsharp=5:5:0.6:3:3:0.2",
            # Subtle push-in motion.
            f"zoompan=z='min(zoom+0.00035,1.05)':d=1:s={WIDTH}x{HEIGHT}:fps={FPS}",
            # Fade in/out to avoid hard cuts and text popping.
            "fade=t=in:st=0:d=0.35",
            f"fade=t=out:st={max(scene.duration_s - 0.45, 0.0):.2f}:d=0.40",
            "format=yuv420p",
        ]
    )


def preprocess_scene(raw_path: Path, out_path: Path, overlay_path: Path, scene: SceneMeta) -> None:
    vf = scene_filter(scene)
    filter_complex = f"[0:v]{vf}[base];[base][1:v]overlay=0:0:format=auto"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-i",
            str(overlay_path),
            "-t",
            str(scene.duration_s),
            "-filter_complex",
            filter_complex,
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )


def concat_clips(clips: list[Path], final_out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        for clip in clips:
            tmp.write(f"file '{clip.resolve().as_posix()}'\n")
        list_path = Path(tmp.name)
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                str(final_out),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble DARWIN cinematic demo")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("media/videos/demo_video"),
        help="Output root used by demo pipeline",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root
    shotlist = root / "shotlist.json"
    raw_dir = root / "raw"
    staged_dir = root / "staged"
    overlay_dir = root / "overlays"
    final_dir = root / "1080p60"
    final_dir.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(shotlist)
    staged_clips: list[Path] = []
    for scene in scenes:
        raw = raw_dir / f"scene_{scene.scene_id}.webm"
        if not raw.exists():
            raise FileNotFoundError(f"Missing raw capture clip: {raw}")
        staged = staged_dir / f"scene_{scene.scene_id}.mp4"
        overlay = overlay_dir / f"scene_{scene.scene_id}.png"
        make_overlay_image(scene, overlay)
        print(f"[assemble] scene {scene.scene_id} -> {staged.name}")
        preprocess_scene(raw, staged, overlay, scene)
        staged_clips.append(staged)

    final_out = final_dir / "DarwinDemo.mp4"
    concat_clips(staged_clips, final_out)
    print(f"[assemble] wrote {final_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

