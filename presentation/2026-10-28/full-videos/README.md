# NeuroChip Twin — Final English Video

## Files

- **NeuroChip_Twin_Presentation_and_QA_EN.mp4** — complete presentation and prepared Q&A appendix. Duration: **5:58** (357.72 seconds). Size:34.7MB.
- **NeuroChip_Twin_Presentation_Only_EN.mp4** — main presentation only. Duration: **4:38** (277.56 seconds). Size:28.1MB.

Both files use1920×1080 Full HD,25fps,H.264 video and AAC48kHz stereo audio. The narration uses the existing ElevenLabs Brian recordings. No new speech was generated and the speaking speed was not changed.

## Structure

The main section combines the ten English scientific animations with the main narration. Transitions follow the existing independent transcript; two boundaries inside mixed-sentence transcript segments are estimated. The original animation MP4s, which contain the same frames as the GIFs with less quantization, are used as the visual sources.

The Q&A appendix begins at **4:38**. Question1 concerns innovation and comparison with interpolation; question2, at **5:21**, concerns real laboratory and organ-on-chip validation. Questions appear on cards and remain visible during their answers. **The questions are displayed, not spoken**; the source audio contains the answers. Five-and-a-half-second silent question cards allow reading before each answer. The appendix includes both additional real-data visuals prepared for questions, so all12 original animations appear in the complete video.

The full MP4 contains chapter markers. Exact timings are available in `chapters.json` and `timeline.json`.

## Validation and provenance

Both files were decoded end to end. Resolution, frame rate, total frame counts, duration, and audio presence were checked. The final decoded audio was compared with the assembled source waveform globally and at representative positions; correlation exceeds0.99975. The question cards' silent audio intervals were checked. Sixteen frames were extracted and inspected for layout, legibility, and question overlays.

`qa/verification.json` records the results. `source_manifest.json` stores input hashes; `render_manifest.json` stores final hashes. All input animations and audio remain unchanged. No repository, scientific report, competition submission, or previously delivered asset was edited. No API credits, GPU computation, third-party uploads, or online publication were used.

The Full HD output scales the1600×900 source animations; it does not add experimental data or new scientific claims. The main video and Q&A reuse the previously checked English wording and calibrated metrics. Machine decoding and waveform comparisons do not constitute human full-length listening.

Build: `python build_video.py`; verify: `python verify_video.py`. The build preserves existing outputs and does not overwrite them. Working segments and assembled WAV audio are kept under `segments/` and `assets/`.
