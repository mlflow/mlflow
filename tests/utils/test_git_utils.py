import pytest

from mlflow.utils.git_utils import _strip_credentials_from_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # HTTPS with username + password (token)
        ("https://user:token@github.com/foo/bar.git", "https://github.com/foo/bar.git"),
        # HTTPS with token-only userinfo (GitHub PAT style)
        ("https://ghp_abc123@github.com/foo/bar.git", "https://github.com/foo/bar.git"),
        # HTTP variant
        ("http://user:pass@example.com/repo.git", "http://example.com/repo.git"),
        # ssh:// URL form with explicit user
        ("ssh://git@github.com/foo/bar.git", "ssh://github.com/foo/bar.git"),
        # git:// URL form with embedded userinfo
        ("git://user:secret@host/repo.git", "git://host/repo.git"),
        # Preserve port
        (
            "https://user:token@host.example.com:8080/repo.git",
            "https://host.example.com:8080/repo.git",
        ),
        # Preserve query + fragment
        ("https://user@host/repo.git?foo=bar#section", "https://host/repo.git?foo=bar#section"),
        # No-op: HTTPS without credentials
        ("https://github.com/foo/bar.git", "https://github.com/foo/bar.git"),
        # No-op: SSH-style scp URL (no scheme to parse)
        ("git@github.com:foo/bar.git", "git@github.com:foo/bar.git"),
        # No-op: local file path
        ("/local/path/to/repo", "/local/path/to/repo"),
        # No-op: empty string
        ("", ""),
    ],
)
def test_strip_credentials_from_url(url: str, expected: str):
    assert _strip_credentials_from_url(url) == expected
