#!/usr/bin/env bash

set -e
./dev/generate-protos.sh

GIT_STATUS="$(git status --porcelain)"
if [ "$GIT_STATUS" ]; then
	echo "Protobufs were not generated with protoc. Please run './dev/generate-protos.sh' or comment '/autoformat' on the PR"
	echo "Git status is"
	echo "------------------------------------------------------------------"
	echo "$GIT_STATUS"
	exit 1
else
	echo "Test successful - Protobuf files generated"
fi
