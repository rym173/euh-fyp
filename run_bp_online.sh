#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}/examples/bp_online"
export PYTHONPATH="${ROOT_DIR}/eoh/src"
exec /opt/anaconda3/bin/python runEoH.py
