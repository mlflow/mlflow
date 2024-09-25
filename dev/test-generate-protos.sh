#!/usr/bin/env bash

set -e
python ./dev/generate_protos.py

GIT_STATUS="$(git status --porcelain)"
if [ "$GIT_STATUS" ]; then 
	echo "Protobufs were not generated with protoc. Please run 'python ./dev/generate_protos.py' or comment '/autoformat' on the PR"
	echo "Git status is"
	echo "------------------------------------------------------------------"
	echo "$GIT_STATUS"
	exit 1
else
	echo "Test successful - Protobuf files generated"
fi
