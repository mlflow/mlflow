"""
Cloudpickle does not include `Unpickler` in its namespace, which is required by
certain ML frameworks (i.e. PyTorch) for deserialization. Adopting the same structure
by which cloudpickle's `compat` module imports `Pickler` from pickle / pickle5 (if available),
this module imports Unpickler from pickle / pickle5
"""

import sys


if sys.version_info < (3, 8):
    try:
        from pickle5 import Unpickler
    except ImportError:
        from pickle import Unpickler
else:
    from _pickle import Unpickler
