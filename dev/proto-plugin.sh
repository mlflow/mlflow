#!/bin/sh

export PYTHONPATH="$PWD:$PYTHONPATH"
exec python dev/proto_plugin.py
