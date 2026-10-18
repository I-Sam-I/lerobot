#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
    echo "Activate the 'lerobot' Conda environment before recording." >&2
    exit 1
fi

exec python "$(dirname "${BASH_SOURCE[0]}")/record_omx.py" record "$@"
