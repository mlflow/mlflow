def strip_prefix(original, prefix):
    if original.startswith(prefix):
        return original[len(prefix):]
    return original


def strip_suffix(original, suffix):
    if original.endswith(suffix) and suffix != '':
        return original[:-len(suffix)]
    return original
