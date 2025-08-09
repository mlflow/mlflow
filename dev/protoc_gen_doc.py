#!/usr/bin/env python
"""Wrapper script to make the proto doc plugin executable by protoc."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proto_plugin import main

if __name__ == "__main__":
    main()
