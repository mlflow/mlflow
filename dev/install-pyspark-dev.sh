#!/usr/bin/env bash

temp_dir=$(mktemp -d)
git clone --depth 1 --branch branch-3.4 https://github.com/apache/spark.git $temp_dir
cd $temp_dir
./build/mvn -DskipTests --no-transfer-progress clean package
cd python
python setup.py bdist_wheel
pip install dist/pyspark-*.whl
