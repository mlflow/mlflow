#!/usr/bin/env bash
set -ex

TEMP_DIR=$(mktemp -d)
git clone --depth 1 --branch branch-3.5 https://github.com/apache/spark.git $TEMP_DIR
cd $TEMP_DIR
./build/mvn -DskipTests --no-transfer-progress clean package
cd python
python setup.py install
