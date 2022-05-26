# Copyright 2018 Databricks, Inc.
import re


VERSION = "1.26.1"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))
