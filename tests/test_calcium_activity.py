import numpy as np
import pytest

from src.calcium_activity import calcium_activity_features


def test_calcium_features_keep_chambers_and_windows_separate():
    times = np.arange(14, dtype=float)
    traces = np.zeros((4, len(times)), dtype=float)
    traces[0, [7, 10]] = [0.8, 0.6]
    traces[1, [7, 10]] = [0.7, 0.5]
    traces[2, [8, 11]] = [0.8, 0.6]
    traces[3, [8, 11]] = [0.7, 0.5]

    result = calcium_activity_features(
        traces,
        times,
        np.array(["treated", "treated", "control", "control"]),
        perturbation_time_s=6,
        background_noise_dff=0.05,
        min_peak_distance_s=2,
    )

    assert list(result["period"]) == ["baseline", "post", "baseline", "post"]
    post = result[result["period"] == "post"].set_index("chamber_id")
    assert set(post.index) == {"treated", "control"}
    assert (post["n_cells"] == 2).all()
    assert (post["event_rate_per_min_mean"] == 15.0).all()
    assert (post["fraction_pairs_above_sync_threshold"] == 1.0).all()
    assert np.allclose(post["mean_event_width_at_10pct_prominence_s"], 1.8)
    assert result.loc[result["period"] == "baseline", "mean_event_prominence_dff"].isna().all()
    assert result.loc[result["period"] == "baseline", "mean_event_width_at_10pct_prominence_s"].isna().all()


def test_calcium_features_reject_irregular_time_sampling():
    times = np.array([0.0, 1.0, 2.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="uniformly sampled"):
        calcium_activity_features(
            np.zeros((2, len(times))),
            times,
            np.array(["a", "b"]),
            perturbation_time_s=3,
            background_noise_dff=0.05,
        )


def test_calcium_features_report_synchrony_as_undefined_for_one_cell():
    traces = np.zeros((1, 8), dtype=float)
    traces[0, 5] = 0.5
    result = calcium_activity_features(
        traces,
        np.arange(8, dtype=float),
        np.array(["single"]),
        perturbation_time_s=4,
        background_noise_dff=0.05,
    )
    assert result["mean_pairwise_correlation"].isna().all()
    assert result["fraction_pairs_above_sync_threshold"].isna().all()


def test_peak_distance_does_not_suppress_events_across_perturbation_boundary():
    times = np.arange(14, dtype=float)
    traces = np.zeros((1, len(times)), dtype=float)
    traces[0, 4] = 0.6
    traces[0, 8] = 0.8

    result = calcium_activity_features(
        traces,
        times,
        np.array(["treated"]),
        perturbation_time_s=7,
        background_noise_dff=0.05,
        min_peak_distance_s=5,
    ).set_index("period")

    assert np.isclose(result.loc["baseline", "event_rate_per_min_mean"], 60 / 7)
    assert np.isclose(result.loc["post", "event_rate_per_min_mean"], 60 / 7)


def test_post_window_cannot_change_baseline_peak_properties():
    times = np.arange(20, dtype=float)
    base = np.zeros((1, len(times)), dtype=float)
    base[0, 8:10] = [0.8, 0.6]
    post_a = base.copy()
    post_a[0, 10:] = [0.7, 0.75, 0.7, 0.72, 0.7, 0.72, 0.7, 0.72, 0.7, 0.72]
    post_b = base.copy()
    post_b[0, 10:] = [0.2, 0.7, 0.2, 0.72, 0.2, 0.72, 0.2, 0.72, 0.2, 0.72]

    def baseline_properties(traces):
        result = calcium_activity_features(
            traces,
            times,
            np.array(["treated"]),
            perturbation_time_s=10,
            background_noise_dff=0.05,
        )
        return result.set_index("period").loc[
            "baseline",
            [
                "event_rate_per_min_mean",
                "mean_event_prominence_dff",
                "mean_event_width_at_10pct_prominence_s",
            ],
        ].to_numpy(dtype=float)

    assert np.allclose(baseline_properties(post_a), baseline_properties(post_b))
