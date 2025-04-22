#!/usr/bin/env bash

DESTINATION="/tmp/typos/bin"
mkdir -p $DESTINATION
wget -q -O - 'https://github.com/crate-ci/typos/releases/download/v1.28.0/typos-v1.28.0-x86_64-unknown-linux-musl.tar.gz' | tar -xz -C $DESTINATION
if [[ $? -eq 0 ]]; then
  chmod +x $DESTINATION/typos
  if [[ $GITHUB_ACTIONS == 'true' ]]; then
    echo $DESTINATION >> $GITHUB_PATH
  fi
else
  # Fall back to cargo install if binary download fails
  cargo install typos-cli@1.28.0 --locked
fi
