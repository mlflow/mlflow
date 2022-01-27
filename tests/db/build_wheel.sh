#!/usr/bin/env bash

set -ex

rm -rf dist
prefix=$(git rev-parse --show-prefix)
pushd $(git rev-parse --show-cdup)
python setup.py bdist_wheel --dist-dir $prefix/dist
popd
