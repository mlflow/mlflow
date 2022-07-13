def strip_prefix(original, prefix):
    if original.startswith(prefix):
        return original[len(prefix) :]
    return original


def strip_suffix(original, suffix):
    if original.endswith(suffix) and suffix != "":
        return original[: -len(suffix)]
    return original


def is_string_type(item):
    return isinstance(item, str)


def truncate_str_from_middle(s, max_length):
    assert max_length > 5
    if len(s) <= max_length:
        return s
    else:
        left_part_len = (max_length - 3) // 2
        right_part_len = max_length - 3 - left_part_len
        return f"{s[:left_part_len]}...{s[-right_part_len:]}"
