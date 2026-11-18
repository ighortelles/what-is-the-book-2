#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_PYTHON="${BOOK_APP_PYTHON:-$REPO_ROOT/.venv-app/bin/python}"
if [[ ! -x "$APP_PYTHON" ]]; then
    echo 'Ambiente ausente. Execute: bash setup_app.sh' >&2
    exit 1
fi
cd "$REPO_ROOT"
export MPLBACKEND=Agg
# Use the bundled models even when a Windows settings.local.json was copied.
# A later --storage argument can explicitly override this presentation default.
exec "$APP_PYTHON" -u -m src.app.server --storage "$REPO_ROOT" "$@"
