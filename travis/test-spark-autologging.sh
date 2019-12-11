#!/usr/bin/env bash

# Test that mlflow installation does not modify (downgrade/upgrade/uninstall) packages from a
# specific Anaconda distribution.

set -eux

git clone https://github.com/apache/spark/tree/master/bin