"""Create a short local MP4 for the Kaggle writeup attachment."""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.neurochip_twin import generate_sequence, phenotype_features


def main() -> None:
    out = Path("outputs/demo/neurochip_twin_demo.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((out.parent / "metrics.json").read_text(encoding="utf-8"))
    calibration = metrics[metrics.get("primary_model", "multimodal_physics")]
    seq = generate_sequence(2026, 0.82, frames=36, scenario="compound_specific")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 8, (960, 600))
    for i, frame in enumerate(seq.frames):
        rgb = np.repeat((frame[..., None] * 255).astype(np.uint8), 3, axis=2)
        image = Image.fromarray(rgb).resize((640, 512)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 36), fill=(16, 24, 40))
        draw.text((14, 10), f"NeuroChip Twin | frame {i+1}/{len(seq.frames)} | compound-specific stress test", fill=(245, 245, 245))
        canvas = Image.new("RGB", (960, 600), (245, 247, 250))
        canvas.paste(image, (20, 20))
        _, table = phenotype_features(replace(seq, frames=seq.frames[: i + 1]))
        last = table[table["frame"] == table["frame"].max()] if not table.empty else table
        draw2 = ImageDraw.Draw(canvas)
        draw2.text((690, 55), "Interpretable state", fill=(20, 30, 50))
        draw2.text((690, 95), f"cells: {len(last)}", fill=(20, 30, 50))
        draw2.text((690, 130), f"dose: {seq.dose:.2f}", fill=(20, 30, 50))
        draw2.text((690, 165), "tracking: Hungarian", fill=(20, 30, 50))
        draw2.text((690, 200), "readout: fixed reservoir", fill=(20, 30, 50))
        draw2.text((690, 235), f"compound context: {seq.compound_id}", fill=(20, 30, 50))
        draw2.text((690, 255), f"Brier/ECE: {calibration['brier_score']:.3f}/{calibration['expected_calibration_error']:.3f}", fill=(20, 30, 50))
        draw2.text((690, 290), "No clinical claim", fill=(160, 45, 40))
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    writer.release()
    print(out)


if __name__ == "__main__":
    main()
