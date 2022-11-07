# Copyright 2018 Databricks, Inc.
import re


VERSION = "2.0.0.dev0"


def foo(a,   b):
    pass


def is_release_version():
    return bool(re.match(r"^\d+\.\d+\.\d+$", VERSION))
