#!/usr/bin/env bash

set -e
python ./dev/generate_protos.sh

GIT_STATUS="$(git status --porcelain)"
if [ "$GIT_STATUS" ]; then
	echo "Protobufs were not generated with protoc. Please run 'python ./dev/generate_protos.sh' or comment '/autoformat' on the PR"
	echo "Git status is"
	echo "------------------------------------------------------------------"
	echo "$GIT_STATUS"
	exit 1
else
	echo "Test successful - Protobuf files generated"
fi
