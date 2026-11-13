#!/usr/bin/env bash
# Pipeline v3 (see run_v3.py): ./run.sh  |  ./run.sh --smoke  |  ./run.sh --summary-only
# Resumable: finished jobs are kept. Models -> models/, report -> results/report.md
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-$HOME/miniconda3/envs/lerobot/bin/python}"
caffeinate -dimsu "$PY" -u run_v3.py "$@" 2>&1 | grep --line-buffered -v -E "WARNING|warnings.warn|XNNPACK|oneDNN"
