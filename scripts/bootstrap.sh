#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/bootstrap.sh --cryosparc-version X.Y
  ./scripts/bootstrap.sh --cryosparc-tools-spec 'cryosparc-tools~=X.Y.0'

Options:
  --cryosparc-version X.Y
      CryoSPARC minor version. This is translated to cryosparc-tools~=X.Y.0.
  --cryosparc-tools-spec SPEC
      Full pip requirement specifier for cryosparc-tools.
  --venv-dir PATH
      Virtual environment path. Default: ./.venv
  --python PYTHON
      Python executable to use for venv creation. Default: python3
  --no-editable
      Install this package normally instead of editable mode.
  -h, --help
      Show this help.

Examples:
  ./scripts/bootstrap.sh --cryosparc-version 5.0
  ./scripts/bootstrap.sh --cryosparc-version 4.7 --venv-dir ~/.venvs/cs-overlay
  ./scripts/bootstrap.sh --cryosparc-tools-spec 'cryosparc-tools~=5.0.0'
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
INSTALL_MODE="-e"
TOOLS_SPEC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cryosparc-version)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --cryosparc-version requires a value" >&2
        exit 1
      fi
      case "$2" in
        [0-9]*.[0-9]*)
          TOOLS_SPEC="cryosparc-tools~=$2.0"
          ;;
        *)
          echo "ERROR: --cryosparc-version must look like 4.7 or 5.0" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --cryosparc-tools-spec)
      TOOLS_SPEC="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-editable)
      INSTALL_MODE=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TOOLS_SPEC}" ]]; then
  echo "ERROR: You must provide either --cryosparc-version or --cryosparc-tools-spec" >&2
  usage >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "${TOOLS_SPEC}"
if [[ -n "${INSTALL_MODE}" ]]; then
  python -m pip install -e "${ROOT_DIR}"
else
  python -m pip install "${ROOT_DIR}"
fi

cat <<EOF

Bootstrap complete.

Next steps:
  source "${VENV_DIR}/bin/activate"
  cryosparc-particle-reposition --help

Installed cryosparc-tools spec:
  ${TOOLS_SPEC}
EOF
