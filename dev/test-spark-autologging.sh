#!/usr/bin/env bash
# Test Spark autologging against the Spark 3.0 preview. This script is temporary and should be
# removed once Spark 3.0 is released in favor of simply updating all tests to run against Spark 3.0
# (i.e. updating the pyspark dependency version in dev/large-requirements.txt)
set -ex

# Build Java package
pushd mlflow/java/spark
mvn package -DskipTests -q
popd

# Install PySpark 3.0 preview & run tests. For faster local iteration, you can also simply download
# the .tgz used below (http://mirror.cogentco.com/pub/apache/spark/spark-3.0.0-preview/spark-3.0.0-preview-bin-hadoop2.7.tgz),
# extract it, and set SPARK_HOME to the path of the extracted folder while invoking pytest as
# shown below
TEMPDIR=$(mktemp -d)
pushd $TEMPDIR
wget --no-verbose https://archive.apache.org/dist/spark/spark-3.0.0-preview/spark-3.0.0-preview-bin-hadoop2.7.tgz -O /tmp/spark.tgz
tar -xf /tmp/spark.tgz
pip install -e spark-3.0.0-preview-bin-hadoop2.7/python
popd

export SPARK_HOME=$TEMPDIR/spark-3.0.0-preview-bin-hadoop2.7
find tests/spark_autologging -name 'test*.py' | xargs -L 1 pytest --large

rm -rf $TEMPDIR

# Reinstall old dependencies
pip install -r dev/large-requirements.txt
