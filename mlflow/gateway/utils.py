def _parse_url_path_for_base_url(url_string):
    split_url = url_string.split("/")
    return "/".join(split_url[:-1])
