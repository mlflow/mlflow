import functools
import hashlib
import sys

# DO NOT use this function for security purposes (e.g., password hashing).
#
# In Python >= 3.9, insecure hashing algorithms such as MD5 fail in FIPS-compliant
# environments unless `usedforsecurity=False` is explicitly passed.
#
# References:
# - https://github.com/mlflow/mlflow/issues/9905
# - https://docs.python.org/3/library/hashlib.html
_kwargs = {"usedforsecurity": False} if sys.version_info >= (3, 9) else {}
md5 = functools.partial(hashlib.md5, **_kwargs)
sha1 = functools.partial(hashlib.sha1, **_kwargs)
