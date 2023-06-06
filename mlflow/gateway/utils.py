import re

RESERVED_CHARACTERS = r"[!\*'();:@&=+\$,/?#\[\]]"


def _parse_url_path_for_base_url(url_string):
    split_url = url_string.split("/")
    return "/".join(split_url[:-1])


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, and hyphen.

    Returns True if the string doesn't contain any of these characters.
    """
    if re.search(RESERVED_CHARACTERS, name) or " " in name or not re.match(r"^[\w-]+$", name):
        return False
    return True
