import numpy as np
import pytest

from safeeyes.data.crop_coverage import coverage_table, crop_coverage, load_row_counts
from safeeyes.data.splits import Sample


def test_single_category_reports_retained_fraction():
    result = coverage_table([("Normal", 100, 25)])
    assert result["per_category"]["Normal"]["feature_rows"] == 100
    assert result["per_category"]["Normal"]["crop_rows"] == 25
    assert result["per_category"]["Normal"]["retained_fraction"] == pytest.approx(0.25)


def test_multiple_samples_in_a_category_sum_rather_than_average():
    # Two samples of the same category with very different sizes. If the
    # fraction were averaged per sample rather than summed then divided, a
    # small sample with a high ratio would pull the category average away
    # from the true pooled rate.
    result = coverage_table(
        [
            ("Talking", 10, 9),  # ratio 0.9
            ("Talking", 990, 1),  # ratio ~0.001
        ]
    )
    entry = result["per_category"]["Talking"]
    assert entry["feature_rows"] == 1000
    assert entry["crop_rows"] == 10
    assert entry["retained_fraction"] == pytest.approx(0.01)


def test_all_aggregates_every_category_not_just_one():
    result = coverage_table(
        [
            ("Normal", 200, 20),
            ("Talking", 100, 70),
            ("Yawning", 100, 60),
        ]
    )
    assert set(result["per_category"]) == {"Normal", "Talking", "Yawning"}
    assert result["all"]["feature_rows"] == 400
    assert result["all"]["crop_rows"] == 150
    assert result["all"]["retained_fraction"] == pytest.approx(150 / 400)


def test_sample_count_is_tracked_per_category():
    result = coverage_table([("Normal", 10, 1), ("Normal", 10, 1), ("Talking", 5, 5)])
    assert result["per_category"]["Normal"]["samples"] == 2
    assert result["per_category"]["Talking"]["samples"] == 1
    assert result["all"]["samples"] == 3


def test_a_category_with_zero_feature_rows_reports_no_fraction_rather_than_dividing_by_zero():
    result = coverage_table([("Normal", 0, 0)])
    assert result["per_category"]["Normal"]["retained_fraction"] is None


def test_crop_rows_may_not_exceed_feature_rows():
    # A crop archive is malformed if it retains more rows than it has features
    # for. This is a contract check on the inputs, not a silent clamp.
    with pytest.raises(ValueError, match="cannot exceed"):
        coverage_table([("Normal", 5, 6)])


# Mutation style checks: these pin the direction and completeness of the
# computation so a future edit that gets either wrong is caught.


def test_ratio_is_crop_rows_over_feature_rows_not_the_other_way_round():
    # A category retaining a small share of its rows must report a small
    # fraction. If the ratio were ever computed inverted (feature_rows over
    # crop_rows) this would report 4.0 instead of 0.25 and fail here.
    result = coverage_table([("Normal", 400, 100)])
    fraction = result["per_category"]["Normal"]["retained_fraction"]
    assert fraction == pytest.approx(0.25)
    assert fraction < 1.0


def test_all_total_would_be_wrong_if_a_category_were_dropped_from_the_denominator():
    # Three categories with distinct, easily separable row counts. The "all"
    # entry must reflect every one of them. A regression that dropped, say,
    # the last category from the aggregation would produce 300/700 (~0.4286)
    # instead of the true 340/1000 (0.34), so this pins the true combined
    # value and would fail under that regression.
    result = coverage_table(
        [
            ("Normal", 200, 20),
            ("Talking", 500, 200),
            ("Yawning", 300, 120),
        ]
    )
    assert result["all"]["feature_rows"] == 1000
    assert result["all"]["crop_rows"] == 340
    assert result["all"]["retained_fraction"] == pytest.approx(0.34)
    dropped_last_category_fraction = 220 / 700
    assert result["all"]["retained_fraction"] != pytest.approx(dropped_last_category_fraction)


def _write_archive(path, n_features, n_crop):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.zeros((n_features, 5), dtype=float),
        frame_indices=np.arange(n_features),
        crop_rows=np.arange(n_crop),
        crops=np.zeros((n_crop, 4, 4, 3), dtype=np.uint8),
    )


def test_load_row_counts_reads_feature_and_crop_row_counts_from_an_archive(tmp_path):
    archive = tmp_path / "a.npz"
    _write_archive(archive, n_features=12, n_crop=5)
    feature_rows, crop_rows = load_row_counts("a", tmp_path)
    assert feature_rows == 12
    assert crop_rows == 5


def test_crop_coverage_walks_samples_by_their_archive_path(tmp_path):
    _write_archive(tmp_path / "Female_mirror" / "1-Normal.npz", n_features=20, n_crop=2)
    _write_archive(tmp_path / "Female_mirror" / "1-Talking.npz", n_features=10, n_crop=8)
    samples = [
        Sample(sample_id="Female_mirror/1-Normal.avi", subject_id="Female1", label="Normal"),
        Sample(sample_id="Female_mirror/1-Talking.avi", subject_id="Female1", label="Talking"),
    ]
    result = crop_coverage(samples, tmp_path)
    assert result["per_category"]["Normal"]["feature_rows"] == 20
    assert result["per_category"]["Normal"]["crop_rows"] == 2
    assert result["per_category"]["Talking"]["feature_rows"] == 10
    assert result["per_category"]["Talking"]["crop_rows"] == 8
    assert result["all"]["feature_rows"] == 30
    assert result["all"]["crop_rows"] == 10
