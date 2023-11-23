name: "py-cache-key"
description: "Generate a key for Python package cache"

outputs:
  key:
    description: "Cache key"
    value: ${{ steps.cache-key.outputs.value }}

runs:
  using: "composite"
  steps:
    - name: cache-key
      id: cache-key
      shell: bash
      env:
        REQUIREMENTS_HASH: ${{ hashFiles('requirements/*requirements.txt') }}
      run: |
        # Refresh cache if the action runner image has changed
        RUNNER_IMAGE="$ImageOS-$ImageVersion"
        # Refresh cache if the python version has changed
        PYTHON_VERSION="$(python --version | cut -d' ' -f2)"
        # Refresh cache daily
        DATE=$(date -u "+%Y%m%d")
        # Change this value to force a cache refresh
        N=1
        echo "value=$RUNNER_IMAGE-$PYTHON_VERSION-$DATE-$REQUIREMENTS_HASH-$N" >> $GITHUB_OUTPUT
