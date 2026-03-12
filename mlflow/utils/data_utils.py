import urllib.parse
from typing import Any


def parse_s3_uri(uri):
    """Parse an S3 URI, returning (bucket, path)"""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3":
        raise Exception(f"Not an S3 URI: {uri}")
    path = parsed.path
    path = path.removeprefix("/")
    return parsed.netloc, path


def is_uri(string):
    parsed_uri = urllib.parse.urlparse(string)
    return len(parsed_uri.scheme) > 0


def is_polars_dataframe(data: Any) -> bool:
    try:
        import polars as pl

        return isinstance(data, pl.DataFrame)
    except ImportError:
        return False
