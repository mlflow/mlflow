def strip_prefix(original, prefix):
    if original.startswith(prefix):
        return original[len(prefix):]
    return original