name: "show-versions"
description: "Show python package versions sorted by release date"
runs:
  using: "composite"
  steps:
    - shell: bash
      run: |
        # Activate the virtual environment if .venv exists
        if [ -d .venv ]; then
          if [ ${{ runner.os }} == "Windows" ]; then
            source .venv/Scripts/activate
          else
            source .venv/bin/activate
          fi
        fi
        pip --disable-pip-version-check install aiohttp > /dev/null
        python dev/show_package_release_dates.py
