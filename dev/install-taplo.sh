#!/usr/bin/env bash

mkdir -p /tmp/taplo/bin
wget -q -O - 'https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-linux-x86_64.gz' | gunzip -c > /tmp/taplo/bin/taplo
if [[ $? -eq 0 ]]; then
  chmod +x /tmp/taplo/bin/taplo
  echo "/tmp/taplo/bin" >> $GITHUB_PATH
else
  # Fall back to cargo install if binary download fails
  cargo install taplo-cli@0.9.3 --locked
fi
