#!/usr/bin/env bash
set -euo pipefail

uv run --project "$(dirname "$0")/.." interaction-labeler run "$@"
