from scripts.make_demo_video import FPS, ROOT, SCENES, load_evidence, write_srt


def test_judge_demo_is_within_competition_video_limit_and_truthful():
    assert sum(scene.seconds for scene in SCENES) < 300
    assert any("synthetically" in scene.caption for scene in SCENES)
    assert any("not toxicity" in scene.caption for scene in SCENES)
    assert any("does not validate organ-on-chip response" in scene.caption for scene in SCENES)
    assert FPS > 0
    evidence = load_evidence(ROOT)
    assert evidence["metrics"]["data_kind"].startswith("synthetic")
    assert evidence["quality"]["n_labelled_images"] == 3072


def test_subtitles_cover_all_demo_scenes(tmp_path):
    path = tmp_path / "demo.srt"
    write_srt(path)
    text = path.read_text(encoding="utf-8")
    assert text.count(" --> ") == len(SCENES)
    assert text.rstrip().endswith(SCENES[-1].caption)
