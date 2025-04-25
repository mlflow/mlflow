# Copyright 2018 Databricks, Inc.
import re

VERSION = "3.0.0rc1"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))
