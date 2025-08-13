#!/usr/bin/env bash

set -euo pipefail

if ! command -v typos &> /dev/null; then
  echo 'typos is not installed. See `dev/typos.md` for how to install it.'
  exit 1
fi

# Make the error message more actionable
typos --format brief --force-exclude --color never "$@" | sed 's/$/. See `dev\/typos.md` for how to fix typos./'
