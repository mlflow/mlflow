# This module is meant to test shadowing of the 3rd party module
raise Exception(
    "This package should not have been imported! "
    "This means that the sys.path was not configured correctly"
)
