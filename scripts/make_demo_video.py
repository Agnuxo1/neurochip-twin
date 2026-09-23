"""Render a captioned, judge-facing NeuroChip Twin walkthrough (under 5 min)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neurochip_twin import generate_sequence, phenotype_features

WIDTH, HEIGHT, FPS = 1280, 720, 12
BG = (8, 17, 31)
PANEL = (18, 33, 55)
PANEL_ALT = (23, 43, 68)
TEXT = (239, 245, 252)
MUTED = (166, 185, 207)
CYAN = (74, 211, 222)
BLUE = (99, 151, 255)
GREEN = (111, 214, 162)
AMBER = (255, 194, 92)
RED = (255, 126, 130)


@dataclass(frozen=True)
class Scene:
    title: str
    subtitle: str
    caption: str
    seconds: int


SCENES = (
    Scene(
        "From microscopy movies to auditable signals",
        "NeuroChip Twin · interpretable temporal analysis for organ-on-chip research",
        "This moving microscopy-like example is generated synthetically; it is not biological evidence.",
        10,
    ),
    Scene(
        "One movie, a traceable analysis path",
        "A compact workflow keeps intermediate measurements visible.",
        "Segment objects, associate them across frames, extract phenotype trajectories, then produce a report.",
        15,
    ),
    Scene(
        "A controlled synthetic stress test",
        "Seed 42 · 180 generated sequences · stratified 25% holdout",
        "The generator includes temporal susceptibility by design. These scores test the software pipeline, not biology.",
        18,
    ),
    Scene(
        "A separate audit on real OoC images",
        "Sample-quality triage · 3,072 labelled images · six held-out cell-line groups",
        "This predicts expert good/bad image quality, not toxicity or treatment response; only six cell-line folds are available.",
        20,
    ),
    Scene(
        "Microscopy portability is a different question",
        "BBBC038 nuclei segmentation audit · 12 calibration / 24 held-out images",
        "BBBC038 measures nuclei segmentation portability only; it does not validate organ-on-chip response prediction.",
        14,
    ),
    Scene(
        "Outputs an experimentalist can inspect",
        "Predictions · phenotype table · calibration · counterfactual · HTML report",
        "The uncertainty display is a distance-from-threshold proxy, not a clinical confidence interval.",
        15,
    ),
    Scene(
        "Reproducible, transparent, and not overclaimed",
        "Public code · CPU-only Kaggle notebook · tests · technical report",
        "No paired neural OoC response data are available; biological validation still requires authorized chip-level experiments.",
        15,
    ),
)


def load_evidence(root: Path = ROOT) -> dict:
    metrics = json.loads((root / "outputs/demo/metrics.json").read_text(encoding="utf-8"))
    quality = json.loads((root / "outputs/ooc_quality_public_summary.json").read_text(encoding="utf-8"))
    bbbc = json.loads((root / "outputs/external_validation_calibrated/bbbc038_summary.json").read_text(encoding="utf-8"))
    if not str(metrics.get("data_kind", "")).startswith("synthetic"):
        raise ValueError("Demo headline metrics must remain explicitly synthetic")
    if "sample image-quality" not in quality.get("task", ""):
        raise ValueError("External OoC audit is not the documented sample-quality task")
    if quality.get("n_labelled_images") != 3072 or len(quality.get("cell_type_counts", {})) != 6:
        raise ValueError("Unexpected OoC sample-quality audit evidence")
    if bbbc.get("task") != "external segmentation portability audit only":
        raise ValueError("BBBC038 must remain scoped to segmentation portability")
    return {"metrics": metrics, "quality": quality, "bbbc": bbbc}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), candidate, font=font)[2] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _paragraph(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont,
               width: int, fill: tuple[int, int, int], line_gap: int = 8) -> int:
    x, y = xy
    for line in _wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += draw.textbbox((0, 0), line, font=font)[3] + line_gap
    return y


def _base(scene_index: int, time_s: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    scene = SCENES[scene_index]
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 82), fill=PANEL)
    draw.text((48, 22), "NEUROCHIP TWIN", font=_font(25, True), fill=CYAN)
    draw.text((WIDTH - 340, 27), "AI4S OPEN INNOVATION", font=_font(18, True), fill=MUTED)
    draw.text((52, 112), f"{scene_index + 1:02d}  /  07", font=_font(16, True), fill=CYAN)
    draw.text((52, 150), scene.title, font=_font(35, True), fill=TEXT)
    _paragraph(draw, scene.subtitle, (54, 205), _font(21), 1120, MUTED, 7)
    draw.line((52, 252, WIDTH - 52, 252), fill=(54, 79, 106), width=2)
    draw.rectangle((0, HEIGHT - 92, WIDTH, HEIGHT), fill=PANEL)
    _paragraph(draw, scene.caption, (48, HEIGHT - 78), _font(17, True), WIDTH - 96, TEXT, 3)
    total = sum(item.seconds for item in SCENES)
    progress = min(1.0, max(0.0, time_s / total))
    draw.rectangle((0, HEIGHT - 8, WIDTH, HEIGHT), fill=(39, 57, 76))
    draw.rectangle((0, HEIGHT - 8, int(WIDTH * progress), HEIGHT), fill=CYAN)
    return canvas, draw


def _pill(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=_font(16, True))
    draw.rounded_rectangle((x - 12, y - 7, box[2] + 12, box[3] + 8), radius=12, fill=color)
    draw.text((x, y), text, font=_font(16, True), fill=BG)


def _metric_bar(draw: ImageDraw.ImageDraw, label: str, value: float, y: int, color: tuple[int, int, int],
                note: str = "") -> None:
    draw.text((76, y), label, font=_font(19, True), fill=TEXT)
    x0, x1, bar_y = 435, 1030, y + 3
    draw.rounded_rectangle((x0, bar_y, x1, bar_y + 23), radius=10, fill=(38, 56, 76))
    draw.rounded_rectangle((x0, bar_y, x0 + int((x1 - x0) * value), bar_y + 23), radius=10, fill=color)
    draw.text((1065, y - 3), f"{value:.3f}", font=_font(20, True), fill=TEXT)
    if note:
        draw.text((76, y + 31), note, font=_font(14), fill=MUTED)


def _draw_frame(scene_index: int, elapsed: int, global_time: int, evidence: dict,
                sequence_frames: list[np.ndarray], sequence_counts: list[int]) -> Image.Image:
    canvas, draw = _base(scene_index, global_time)
    if scene_index == 0:
        frame_index = min(len(sequence_frames) - 1, elapsed * len(sequence_frames) // SCENES[0].seconds)
        frame = Image.fromarray(sequence_frames[frame_index]).resize((600, 390))
        canvas.paste(frame, (54, 280))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((54, 280, 654, 670), radius=12, outline=CYAN, width=3)
        _pill(draw, "SYNTHETIC · DISPLAY CONTRAST ×3", (78, 300), AMBER)
        draw.text((730, 330), "Time-lapse input", font=_font(27, True), fill=TEXT)
        draw.text((730, 390), f"Frame {frame_index + 1:02d} / {len(sequence_frames)}", font=_font(21), fill=MUTED)
        draw.text((730, 435), f"Detected objects: {sequence_counts[frame_index]}", font=_font(21), fill=CYAN)
        draw.text((730, 480), "Dose context: 0.82 (simulated)", font=_font(19), fill=MUTED)
        draw.text((730, 540), "No biological sample is shown", font=_font(18, True), fill=AMBER)

    elif scene_index == 1:
        steps = [
            ("01", "Image sequence", "input movie", CYAN),
            ("02", "Segmentation", "objects / masks", BLUE),
            ("03", "Association", "Hungarian tracks", GREEN),
            ("04", "Phenotypes", "shape · motion · intensity", AMBER),
            ("05", "Readout", "risk + uncertainty proxy", RED),
        ]
        x0, gap, box_w = 64, 232, 204
        for idx, (number, title, detail, color) in enumerate(steps):
            x = x0 + idx * gap
            draw.rounded_rectangle((x, 330, x + box_w, 485), radius=17, fill=PANEL_ALT, outline=color, width=3)
            draw.text((x + 17, 350), number, font=_font(17, True), fill=color)
            draw.text((x + 17, 392), title, font=_font(19, True), fill=TEXT)
            _paragraph(draw, detail, (x + 17, 431), _font(14), box_w - 30, MUTED, 4)
            if idx < len(steps) - 1:
                draw.line((x + box_w + 5, 408, x + gap - 14, 408), fill=MUTED, width=3)
                draw.polygon([(x + gap - 22, 401), (x + gap - 12, 408), (x + gap - 22, 415)], fill=MUTED)
        draw.text((70, 550), "Model design", font=_font(18, True), fill=CYAN)
        draw.text((235, 550), "fixed temporal reservoir + compact regularized multimodal readout", font=_font(19), fill=TEXT)

    elif scene_index == 2:
        metrics = evidence["metrics"]
        values = [
            ("Static baseline", metrics["baseline"]["roc_auc"], MUTED),
            ("Physics-only", metrics["physics_only"]["roc_auc"], BLUE),
            ("Temporal reservoir", metrics["temporal_reservoir"]["roc_auc"], GREEN),
            ("Additive multimodal", metrics["multimodal_no_interactions"]["roc_auc"], CYAN),
        ]
        for idx, (label, value, color) in enumerate(values):
            _metric_bar(draw, label, float(value), 288 + idx * 76, color)
        _pill(draw, "SYNTHETIC BENCHMARK ONLY", (78, 572), AMBER)
        draw.text((430, 580), "Generator encodes temporal susceptibility by design.", font=_font(17), fill=MUTED)

    elif scene_index == 3:
        quality = evidence["quality"]
        models = quality["models"]
        display = [
            ("HOG image features", models["image_hog"]["cell_type_macro_roc_auc"], MUTED),
            ("HOG + metadata", models["image_plus_metadata"]["cell_type_macro_roc_auc"], BLUE),
            ("Frozen Inception-v3", models["inception_v3_logistic"]["cell_type_macro_roc_auc"], GREEN),
            ("Inception-v3 + metadata", models["inception_v3_plus_metadata"]["cell_type_macro_roc_auc"], CYAN),
        ]
        for idx, (label, value, color) in enumerate(display):
            _metric_bar(draw, label, float(value), 282 + idx * 62, color)
        counts = quality["cell_type_counts"]
        detail = "  ·  ".join(f"{name}: {n}" for name, n in counts.items())
        draw.text((76, 548), f"Image counts by line: {detail}", font=_font(14), fill=MUTED)
        _pill(draw, "QUALITY LABELS ≠ RESPONSE LABELS", (78, 595), AMBER)
        draw.text((500, 603), "6 groups only; HUVEC has 15 good / 92 bad images.", font=_font(16), fill=MUTED)

    elif scene_index == 4:
        metrics = evidence["bbbc"]["metrics"]
        cards = [
            ("Pixel IoU", float(metrics["iou"]["mean"]), BLUE),
            ("Pixel Dice", float(metrics["dice"]["mean"]), GREEN),
            ("Precision", float(metrics["precision"]["mean"]), CYAN),
            ("Recall", float(metrics["recall"]["mean"]), AMBER),
        ]
        for i, (label, value, color) in enumerate(cards):
            x = 76 + (i % 2) * 575
            y = 318 + (i // 2) * 138
            draw.rounded_rectangle((x, y, x + 515, y + 106), radius=18, fill=PANEL_ALT, outline=color, width=2)
            draw.text((x + 24, y + 20), label, font=_font(20, True), fill=MUTED)
            draw.text((x + 365, y + 13), f"{value:.3f}", font=_font(35, True), fill=color)
        draw.text((80, 594), "24 held-out microscopy images  ·  12 separate calibration images", font=_font(17), fill=TEXT)

    elif scene_index == 5:
        paths = [
            ("outputs/demo/predictions.csv", "cell-level prediction table"),
            ("outputs/demo/counterfactual_flow.csv", "flow scenario comparison"),
            ("outputs/demo/calibration_curve.png", "probability calibration check"),
            ("outputs/demo/index.html", "shareable local result summary"),
        ]
        for i, (path, description) in enumerate(paths):
            y = 300 + i * 72
            draw.rounded_rectangle((72, y, 1200, y + 52), radius=12, fill=PANEL_ALT)
            draw.text((98, y + 13), path, font=_font(17, True), fill=CYAN)
            draw.text((655, y + 13), description, font=_font(17), fill=TEXT)
        draw.text((82, 612), "Headline metrics include reproduction commands in the technical report.", font=_font(17), fill=MUTED)

    elif scene_index == 6:
        bullets = [
            ("Reproduce", "python -m src.neurochip_twin --out outputs/demo --seed 42 --samples 180 --scenario compound_specific"),
            ("Test", "python -m pytest -q"),
            ("Inspect", "README · technical report · Kaggle CPU notebook · exact data/licence notes"),
        ]
        for i, (label, detail) in enumerate(bullets):
            y = 305 + i * 88
            draw.ellipse((82, y + 4, 116, y + 38), fill=GREEN)
            draw.text((138, y), label, font=_font(21, True), fill=GREEN)
            _paragraph(draw, detail, (300, y), _font(17), 850, TEXT, 4)
        draw.rounded_rectangle((78, 576, 1198, 653), radius=15, fill=(62, 48, 30), outline=AMBER, width=2)
        _paragraph(draw, "Not a clinical or dosing tool. No paired neural organ-on-chip response validation is available yet.",
                   (100, 594), _font(17, True), 1065, AMBER, 4)

    return canvas


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(path: Path) -> None:
    cursor = 0
    entries = []
    for index, scene in enumerate(SCENES, start=1):
        end = cursor + scene.seconds
        entries.append(f"{index}\n{_srt_time(cursor)} --> {_srt_time(end)}\n{scene.caption}\n")
        cursor = end
    path.write_text("\n".join(entries), encoding="utf-8")


def render(out: Path, root: Path = ROOT) -> tuple[Path, Path]:
    evidence = load_evidence(root)
    sequence = generate_sequence(2026, 0.82, frames=36, scenario="compound_specific")
    _, phenotype_table = phenotype_features(sequence)
    # A fixed display-only gain makes the dim synthetic objects visible; the
    # quantitative pipeline still receives the untouched generator frames.
    sequence_frames = [np.repeat((np.clip(frame * 3.0, 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2) for frame in sequence.frames]
    sequence_counts = [int((phenotype_table["frame"] == idx).sum()) for idx in range(len(sequence.frames))]

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out}")
    try:
        total_frames = sum(scene.seconds for scene in SCENES) * FPS
        frame_no = 0
        for scene_index, scene in enumerate(SCENES):
            for elapsed in range(scene.seconds * FPS):
                second = elapsed // FPS
                image = _draw_frame(scene_index, second, frame_no // FPS, evidence, sequence_frames, sequence_counts)
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                frame_no += 1
                if frame_no % (FPS * 15) == 0:
                    print(f"Rendered {frame_no}/{total_frames} frames")
    finally:
        writer.release()
    subtitles = out.with_suffix(".srt")
    write_srt(subtitles)
    return out, subtitles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/demo/neurochip_twin_judges_demo.mp4")
    args = parser.parse_args()
    video, subtitles = render(args.out)
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    duration = frames / fps if fps else 0
    print(f"Video: {video} ({duration:.1f}s, {frames} frames, {fps:g} fps)")
    print(f"Captions: {subtitles}")
    if not duration or duration >= 300:
        raise RuntimeError("Judge demo must be playable and shorter than the 5-minute limit")


if __name__ == "__main__":
    main()
