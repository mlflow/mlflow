import urllib.parse


def parse_s3_uri(uri):
    """Parse an S3 URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3":
        raise Exception(f"Not an S3 URI: {uri}")
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]
    return parsed.netloc, path


def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0
