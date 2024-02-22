#!/usr/bin/env bash

if ! command -v taplo &> /dev/null
then
  echo "taplo could not be found."
  echo "Please install taplo using one of the following methods:"
  echo "- homebrew: https://taplo.tamasfe.dev/cli/installation/homebrew.html"
  echo "- cargo: https://taplo.tamasfe.dev/cli/installation/cargo.html"
  echo "- npm: https://taplo.tamasfe.dev/cli/installation/npm.html"
  echo "- vscode: https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml"
  exit 1
fi

taplo format "$@"
