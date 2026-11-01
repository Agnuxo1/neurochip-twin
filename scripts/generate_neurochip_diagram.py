"""Create a clearly conceptual README animation for NeuroChip Twin v2.

This diagram conveys the published research workflow without reproducing biological
recordings, measurements, fitted predictions or leaderboard results. It contains
no experimental data, synthetic scores, or misleading uncertainty intervals.
"""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WIDTH = 800
HEIGHT = 240
FRAMES = 18
BACKGROUND = "#111b2d"
PANEL = "#22334d"
BORDER = "#506b85"
WHITE = "#eff6fa"
MUTED = "#b1c5d6"
CYAN = "#72d6dc"
GOLD = "#e9c980"


def frame(index: int) -> Image.Image:
    """Draw one accessible and explanatory frame of the conceptual pipeline."""
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(canvas)
    d.text((24, 12), "NEUROCHIP TWIN V2", fill=WHITE)
    d.text((24, 34), "Conceptual workflow: few-shot neural-network response forecasting", fill=MUTED)
    blocks = [
        (24, "MEA assay", "Measured signals"),
        (225, "Few-shot context", "Observed dose/time"),
        (426, "Neural process", "Conditional encoder"),
        (627, "Forecast", "Response + uncertainty"),
    ]
    for x, label, sublabel in blocks:
        d.rounded_rectangle((x, 70, x + 164, 176), radius=14, fill=PANEL, outline=BORDER, width=2)
        d.text((x + 13, 81), label, fill=WHITE)
        d.text((x + 13, 101), sublabel, fill=MUTED)
    phase = index / FRAMES
    for j, x in enumerate((188, 389, 590)):
        d.line((x + 7, 121, x + 32, 121), fill=CYAN, width=3)
        d.polygon(((x + 32, 121), (x + 24, 115), (x + 24, 127)), fill=CYAN)
        pulse_x = x + 7 + (index * 3 + j * 8) % 26
        d.ellipse((pulse_x - 3, 118, pulse_x + 3, 124), fill=GOLD)
    for bar in range(8):
        x = 35 + bar * 17
        amp = 13 + 6 * math.sin(phase * math.tau + bar * 0.6)
        d.line((x, 152, x, 152 - amp), fill=CYAN, width=3)
    for k in range(3):
        d.ellipse((242 + k * 39, 137, 250 + k * 39, 145), fill=GOLD)
    for layer, total in ((452, 3), (496, 4), (541, 3)):
        for n in range(total):
            d.ellipse((layer, 127 + 12 * n, layer + 6, 133 + 12 * n), fill=CYAN)
    for x in range(652, 760):
        y = int(143 - 13 * math.sin((x - 652) / 27))
        d.point((x, y), fill=GOLD)
        d.point((x, y - 9), fill=BORDER)
        d.point((x, y + 9), fill=BORDER)
    d.rounded_rectangle((24, 197, 776, 204), radius=4, fill="#37475b")
    d.rounded_rectangle((24, 197, 24 + int(752 * (index + 1) / FRAMES), 204), radius=4, fill=CYAN)
    d.text((24, 215), "CONCEPTUAL SCHEMATIC  -  NOT RAW DATA, MODEL OUTPUT, OR MEASURED PERFORMANCE", fill=MUTED)
    return canvas


def main() -> None:
    """Generate the animated GIF and an exact-size static PNG alternative."""
    destination = Path(__file__).resolve().parent
    frames = [frame(i) for i in range(FRAMES)]
    frames[FRAMES // 2].save(destination / "neurochip-concept.png", optimize=True)
    frames[0].save(
        destination / "neurochip-concept.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        loop=0,
        duration=125,
    )
    gif = Image.open(destination / "neurochip-concept.gif")
    png = Image.open(destination / "neurochip-concept.png")
    assert gif.n_frames == FRAMES, "All conceptual animation frames must be present."
    assert gif.size == png.size == (WIDTH, HEIGHT)
    print(f"Verified {gif.n_frames} GIF frames and PNG resolution {png.size}")


if __name__ == "__main__":
    main()
