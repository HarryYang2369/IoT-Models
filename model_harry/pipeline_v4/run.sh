#!/usr/bin/env bash
# Pipeline v4 (see run_v4.py and README.md). Works on macOS and on Linux / WSL2 (Windows laptop, see
# WINDOWS_GPU_GUIDE.md).
#   ./run.sh                          all jobs (resumable: finished jobs are kept)
#   ./run.sh --folds 5,6,7,8,9 --gpu  only these folds, on the NVIDIA GPU
#   ./run.sh --smoke [--gpu]          1-epoch end-to-end test in results_smoke/
#   ./run.sh --summary-only           rebuild results/report.md from the finished jobs
set -uo pipefail
cd "$(dirname "$0")"
if [[ -z "${PY:-}" ]]; then
  for c in ../../.venv/bin/python ../.venv/bin/python "$HOME/miniconda3/envs/lerobot/bin/python" python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import tensorflow, pandas, sklearn" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[[ -z "${PY:-}" ]] && { echo "No Python with tensorflow, pandas and scikit-learn found (see WINDOWS_GPU_GUIDE.md)." >&2; exit 1; }
echo "Python: $PY"
if [[ " $* " != *" --smoke "* && " $* " != *" --summary-only "* ]]; then
  while pgrep -f "run_v3.py" >/dev/null; do echo "$(date '+%H:%M') waiting for pipeline v3 to finish ..."; sleep 120; done
fi
KEEP_AWAKE=()
command -v caffeinate >/dev/null 2>&1 && KEEP_AWAKE=(caffeinate -dimsu)      # macOS: no sleep while running
"${KEEP_AWAKE[@]}" "$PY" -u run_v4.py "$@" 2>&1 | grep --line-buffered -v -E "WARNING|warnings.warn|XNNPACK|oneDNN"
