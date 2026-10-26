#!/usr/bin/env bash
# Runs the three services for local development (no Docker). Needs: .env (copy .env.example), Python deps in ml-service/requirements.txt
# plus torch/torchvision, Node >= 22.13 with `npm install` done in server/ and web/. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Create .env first: cp .env.example .env and fill in the secrets"; exit 1; }
set -a; . ./.env; set +a
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-$PWD/${ARTIFACTS_PATH:-artifacts}}"
export ML_SERVICE_URL="http://127.0.0.1:8000" DATA_DIR="$PWD/data" PORT=4000 TQDM_DISABLE=1
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT INT TERM
(cd ml-service && python -m uvicorn app.main:app --port 8000) & pids+=($!)
(cd server && node --disable-warning=ExperimentalWarning --watch src/index.js) & pids+=($!)
(cd web && npx vite --host) & pids+=($!)
echo "Open http://localhost:5173 (the ML service needs about a minute to load the networks)"
wait
