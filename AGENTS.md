# Contributor and agent guide

This is a file-based CPU renderer, not a CryoSPARC server/API client. Read README.md
for supported job types, installation, configuration precedence, and storage rules.

- Python >=3.10; install with `python -m pip install -e ".[dev]"`.
- Rendering requires an independently installed `cryosparc-tools` matching the
  user's CryoSPARC minor release: https://tools.cryosparc.com/#installation.
  Helper/config tests do not require CryoSPARC or private data.
- Run `python -m pytest` and `python -m cryosparc_2d_class_overlay --help`.
- `src/cryosparc_2d_class_overlay/cli.py` owns dataset loading and rendering;
  `config.py` translates JSON into the CLI parser's validated settings.
- Preserve all three console entry points in pyproject.toml and the Python module
  name; the preferred command is `cryosparc-particle-reposition`.
- Keep institutional paths, credentials, hostnames, scheduler defaults, and
  environment names out of source and examples. Use generic placeholders.
- Never infer a cstools path or modify a CryoSPARC server environment. Use the
  active Python interpreter. Input datasets are read-only; outputs belong in the
  requested directory. Do not rewrite `.cs` paths or scientific transforms as a
  side effect of installation/configuration changes.
- When changing options, update README/examples and test CLI/config precedence,
  validation, and path semantics. Use synthetic temporary fixtures, not lab data.
- Real rendering validation requires a supported job and its referenced files;
  clearly distinguish helper tests from integration validation.
