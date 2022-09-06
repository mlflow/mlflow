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


def generate_feature_name_if_not_string(s):
    if isinstance(s, str):
        return s

    return f"feature_{s}"


def truncate_str_from_middle(s, max_length):
    assert max_length > 5
    if len(s) <= max_length:
        return s
    else:
        left_part_len = (max_length - 3) // 2
        right_part_len = max_length - 3 - left_part_len
        return f"{s[:left_part_len]}...{s[-right_part_len:]}"


def dedup_string_list(str_list):
    """
    De-duplicate a list of strings. For duplicated strings, add suffix such as (1), (2), etc.
    """
    count_dict = {}
    dedup_list = []
    for s in str_list:
        if s not in count_dict:
            count_dict[s] = 1
            new_s = s
        else:
            count_dict[s] += 1
            new_s = f"{s}({count_dict[s]})"
        dedup_list.append(new_s)
    return dedup_list
