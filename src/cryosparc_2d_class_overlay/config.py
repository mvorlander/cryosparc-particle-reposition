"""Translate explicit JSON configuration into validated command-line options."""

import argparse
import json
import sys
from pathlib import Path


PATH_KEYS = {"job_dir", "job_dir_2", "denoise_job_dir", "output_dir"}


def parse_arguments(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Help/version must remain usable without input data or a readable config.
    if any(flag in argv for flag in ("-h", "--help", "--version")):
        return parser.parse_args(argv)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--config")
    config_path = probe.parse_known_args(argv)[0].config
    if not config_path:
        return parser.parse_args(argv)
    path = Path(config_path).expanduser().resolve()
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read config {path}: {exc}")
    if not isinstance(settings, dict):
        parser.error("Config must be a JSON object")

    actions = {
        action.dest: action for action in parser._actions
        if action.dest not in {"help", "version", "config"}
    }
    explicit = {
        action.dest for action in parser._actions
        if any(token.split("=", 1)[0] in action.option_strings for token in argv)
    }
    tokens = []
    for key, value in settings.items():
        if key not in actions:
            parser.error(f"Unknown config key: {key}")
        action = actions[key]
        option = action.option_strings[0]
        values = value if isinstance(action, argparse._AppendAction) else [value]
        if not isinstance(values, list) or not values:
            parser.error(f"Config {key} must be a nonempty JSON array")
        configured = []
        for item in values:
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(item, bool):
                    parser.error(f"Config {key} must be a boolean")
                # Set booleans as defaults, so explicit CLI switches take precedence.
                if key not in explicit:
                    parser.set_defaults(**{key: item})
                continue
            expected = action.type or str
            if (expected is int and type(item) is not int or
                    expected is float and type(item) not in (int, float) or
                    expected is str and not isinstance(item, str)):
                parser.error(f"Config {key} has an invalid value type")
            if action.choices is not None and item not in action.choices:
                parser.error(f"Config {key} must be one of {action.choices}")
            if key in PATH_KEYS:
                item = Path(item).expanduser()
                item = str(item if item.is_absolute() else path.parent / item)
            configured.append(f"{option}={item}")
        if key not in explicit:
            tokens.extend(configured)
    return parser.parse_args(tokens + argv)
