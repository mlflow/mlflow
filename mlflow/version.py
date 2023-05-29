# Copyright 2018 Databricks, Inc.
import re

#TODO HACKED REMOVE!
VERSION = "2.2.26"


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))
