from pathlib import Path

import numpy as np

from cryosparc_2d_class_overlay.cli import (
    OverlaySource,
    SOURCE_KIND_REFINE3D,
    SOURCE_KIND_SELECT2D,
    default_overlay_colors,
    find_latest_job_file,
    harmonic_mean,
    normalize_field_label,
    normalize_source_colors,
    normalize_source_subsets,
    parse_rgb_color,
    quantize_pose3d,
    rank_micrograph_items,
    resolve_synthetic_background_color,
)


def test_normalize_field_label():
    assert normalize_field_label("J46") == "J46"
    assert normalize_field_label("J 46 / selected") == "J_46_selected"


def test_harmonic_mean():
    assert harmonic_mean([1.0, 1.0]) == 1.0
    assert harmonic_mean([2.0, 4.0]) == 2.6666666666666665
    assert harmonic_mean([0.0, 1.0]) == 0.0


def test_resolve_synthetic_background_color_auto_uses_white_for_dark_overlay():
    background = resolve_synthetic_background_color(
        "auto",
        [parse_rgb_color("black")],
    )
    assert np.allclose(background, parse_rgb_color("white"))


def test_default_overlay_colors_extend_beyond_two_sources():
    colors = default_overlay_colors(4)
    assert colors == ["black", "red", "cyan", "yellow"]


def test_normalize_source_subsets_repeats_single_value():
    assert normalize_source_subsets(3, ["excluded"], None) == [
        "excluded",
        "excluded",
        "excluded",
    ]


def test_normalize_source_colors_overrides_first_two_then_fills_defaults():
    colors = normalize_source_colors(4, ["white"], "lime")
    assert colors == ["white", "lime", "cyan", "yellow"]


def test_rank_micrograph_items_balanced_prefers_shared_signal():
    sources = [
        OverlaySource("J46", Path("/tmp/J46"), "selected", SOURCE_KIND_SELECT2D, "black", parse_rgb_color("black"), {}, None, {}),
        OverlaySource("J98", Path("/tmp/J98"), "selected", SOURCE_KIND_SELECT2D, "red", parse_rgb_color("red"), {}, None, {}),
    ]
    items = [
        (Path("mic_a.mrc"), [list(range(80)), list(range(2))]),
        (Path("mic_b.mrc"), [list(range(12)), list(range(12))]),
        (Path("mic_c.mrc"), [list(range(20)), list(range(9))]),
    ]
    ranked = rank_micrograph_items(items, sources, "balanced", 3)
    assert ranked[0][0].name == "mic_c.mrc"


def test_find_latest_job_file_prefers_latest_iteration(tmp_path: Path):
    job_dir = tmp_path / "J95"
    job_dir.mkdir()
    (job_dir / "J95_001_particles.cs").write_text("")
    (job_dir / "J95_004_particles.cs").write_text("")
    (job_dir / "J95_particles.cs").write_text("")
    assert find_latest_job_file(job_dir, "particles.cs").name == "J95_004_particles.cs"


def test_rank_micrograph_items_sum_allows_mixed_source_kinds():
    sources = [
        OverlaySource("J46", Path("/tmp/J46"), "selected", SOURCE_KIND_SELECT2D, "black", parse_rgb_color("black"), {}, None, {}),
        OverlaySource("J95", Path("/tmp/J95"), None, SOURCE_KIND_REFINE3D, "red", parse_rgb_color("red"), None, None, {}),
    ]
    items = [
        (Path("mic_a.mrc"), [list(range(10)), list(range(3))]),
        (Path("mic_b.mrc"), [list(range(7)), list(range(8))]),
    ]
    ranked = rank_micrograph_items(items, sources, "sum", 2)
    assert ranked[0][0].name == "mic_b.mrc"


def test_quantize_pose3d_zero_step_keeps_pose():
    pose = (0.12, -0.34, 0.56)
    quantized = quantize_pose3d(pose, 0.0)
    assert np.allclose(quantized, pose)
