"""
Backward-compatibility entry point that delegates to `flavors update`.

Internal jobs still invoke `python dev/update_ml_package_versions.py [--skip-yml]`.
Prefer `uv run flavors update` for new callers.

TODO: Delete this file once all internal jobs have migrated.
"""

import sys

from flavors._cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "update", *sys.argv[1:]]
    main()
