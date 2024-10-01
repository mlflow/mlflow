#!/usr/bin/env bash

DST="/tmp/taplo/bin"
mkdir -p $DST
wget -q -O - 'https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-linux-x86_64.gz' | gunzip -c > $DST/taplo
if [[ $? -eq 0 ]]; then
  chmod +x $DST/taplo
  echo $DST >> $GITHUB_PATH
else
  # Fall back to cargo install if binary download fails
  cargo install taplo-cli@0.9.3 --locked
fi
