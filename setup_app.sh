#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo 'Instale uv: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
if [[ ! -x .venv-app/bin/python ]]; then
    uv venv --python 3.13 .venv-app
fi
uv pip install --python .venv-app/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu'
uv pip install --python .venv-app/bin/python -r requirements-app.txt
.venv-app/bin/python -c 'import cv2, torch, ultralytics; print("Ambiente pronto:", torch.__version__, ultralytics.__version__)'
echo 'Para iniciar: bash start_app.sh'
