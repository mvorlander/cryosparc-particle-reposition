import json
from pathlib import Path

import pytest

from cryosparc_2d_class_overlay.cli import build_argument_parser
from cryosparc_2d_class_overlay.config import parse_arguments


def parse(tmp_path, settings, *options):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return parse_arguments(build_argument_parser(), ["--config", str(path), *options])


def test_config_paths_and_types(tmp_path):
    args = parse(tmp_path, {"job_dir": ["project/J1"], "output_dir": "results",
                            "overlay_color": ["red"], "top_micrographs": 2,
                            "write_gifs": False, "class_opacity": 0.5})
    assert args.job_dir == [str(tmp_path / "project/J1")]
    assert args.output_dir == str(tmp_path / "results")
    assert args.top_micrographs == 2
    assert args.class_opacity == 0.5
    assert args.write_gifs is False


def test_cli_replaces_arrays_and_booleans(tmp_path):
    args = parse(tmp_path, {"job_dir": ["J1", "J2"], "write_gifs": False,
                            "top_micrographs": 5, "overlay_color": ["red", "blue"]},
                 "--job-dir=local/J3", "--job-dir", "local/J4", "--write-gifs",
                 "--top-micrographs", "1", "--overlay-color", "cyan")
    assert args.job_dir == ["local/J3", "local/J4"]
    assert args.overlay_color == ["cyan"]
    assert args.write_gifs is True
    assert args.top_micrographs == 1


@pytest.mark.parametrize("settings", [[], {"typo": 1}, {"job_dir": "J1"},
    {"job_dir": []}, {"job_dir": [1]}, {"top_micrographs": "2"},
    {"top_micrographs": True}, {"write_gifs": "false"}, {"subset": ["invalid"]},
    {"config": "nested.json"}])
def test_invalid_config(tmp_path, settings):
    with pytest.raises(SystemExit) as exc:
        parse(tmp_path, settings, "--job-dir", "J1")
    assert exc.value.code == 2


def test_missing_and_malformed_config(tmp_path):
    path = tmp_path / "bad.json"
    for content in (None, "{broken"):
        if content:
            path.write_text(content)
        with pytest.raises(SystemExit) as exc:
            parse_arguments(build_argument_parser(), ["--config", str(path)])
        assert exc.value.code == 2


def test_required_job_and_help(tmp_path):
    with pytest.raises(SystemExit) as exc:
        parse(tmp_path, {})
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        parse_arguments(build_argument_parser(), ["--config", "missing.json", "--help"])
    assert exc.value.code == 0


def test_cli_without_config():
    args = parse_arguments(build_argument_parser(), ["--job-dir", "J1", "--no-write-gifs"])
    assert args.job_dir == ["J1"]
    assert not args.write_gifs
