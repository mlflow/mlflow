#!/bin/bash
set -ex
if ! [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  ./travis/test-anaconda-compatibility.sh "anaconda3:2020.02"
  ./travis/test-anaconda-compatibility.sh "anaconda3:2019.03"
fi
