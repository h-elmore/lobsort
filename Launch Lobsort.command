#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/lobsort ]; then
  echo "Setting up Lobsort. The first launch needs an internet connection."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e .
fi
exec .venv/bin/lobsort "$@"
