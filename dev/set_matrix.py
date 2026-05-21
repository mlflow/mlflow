"""
Backward-compatibility wrapper around `flavors matrix`.

Prefer `uv run flavors matrix ...`. This script exists for internal jobs that
still invoke `python dev/set_matrix.py ...`; it will be removed once they
migrate.

TODO: Delete this file once all internal jobs have migrated to
`flavors matrix`.
"""

import asyncio
import sys

# Re-exported for legacy importers (tests, internal scripts).
from flavors._matrix import (  # noqa: F401
    MatrixItem,
    apply_changed_files,
    expand_config,
    filter_versions,
    generate_matrix,
    get_changed_flavors,
    get_latest_micro_versions,
    parse_args,
    run,
)

if __name__ == "__main__":
    asyncio.run(run(parse_args(sys.argv[1:])))
