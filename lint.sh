#!/usr/bin/env bash

set -e

FWDIR="$(cd "`dirname $0`"; pwd)"
cd "$FWDIR"

pylint --rcfile="$FWDIR/pylintrc" mlflow

rstcheck README.rst
