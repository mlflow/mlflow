#!/usr/bin/env bash

if ! command -v taplo &> /dev/null
then
  echo "taplo could not be found."
  echo "Install taplo using the following instructions:"
  echo "- homebrew: https://taplo.tamasfe.dev/cli/installation/homebrew.html"
  echo "- cargo: https://taplo.tamasfe.dev/cli/installation/cargo.html"
  echo "- npm: https://taplo.tamasfe.dev/cli/installation/npm.html"
  exit 1
fi

taplo format "$@"
