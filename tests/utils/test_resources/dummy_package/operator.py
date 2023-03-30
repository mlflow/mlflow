# This module is meant to test shadowing of the built-in operator module
raise Exception(
    "This package should not have been imported! "
    "This means that the sys.path was not configured correctly"
)
