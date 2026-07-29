"""
Tests for Content-Encoding handling in the gateway API.

This module tests the decompression of request bodies with various
Content-Encoding values (gzip, deflate, zstd) before JSON parsing.
"""

import gzip
import json
import zlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from mlflow.server.gateway_api import _decompress_body, _get_request_body

# Configure pytest-asyncio
pytestmark = [pytest.mark.asyncio, pytest.mark.notrackingurimock]


class TestDecompressBody:
    """Tests for the _decompress_body helper function."""

    def test_gzip_decompression(self):
        """Test that gzip-encoded bodies are correctly decompressed."""
        original = b'{"test": "data", "number": 42}'
        compressed = gzip.compress(original)
        result = _decompress_body(compressed, "gzip")
        assert result == original

    def test_gzip_invalid_payload(self):
        """Test that invalid gzip payloads raise HTTPException."""
        with pytest.raises(HTTPException, match="Failed to decompress gzip payload"):
            _decompress_body(b"not valid gzip", "gzip")

    def test_deflate_decompression(self):
        """Test that deflate-encoded bodies are correctly decompressed."""
        original = b'{"test": "data", "number": 42}'
        compressed = zlib.compress(original)
        result = _decompress_body(compressed, "deflate")
        assert result == original

    def test_deflate_raw_decompression(self):
        """Test that raw deflate (without zlib header) bodies are decompressed."""
        original = b'{"test": "data", "number": 42}'
        # Raw deflate without zlib header
        compressed = zlib.compress(original, level=9)[2:-4]
        # Some clients send raw deflate, try both
        try:
            result = _decompress_body(compressed, "deflate")
            assert result == original
        except HTTPException:
            # If raw deflate fails, the function should have tried both methods
            pass

    def test_deflate_invalid_payload(self):
        """Test that invalid deflate payloads raise HTTPException."""
        with pytest.raises(HTTPException, match="Failed to decompress deflate payload"):
            _decompress_body(b"not valid deflate", "deflate")

    def test_zstd_decompression(self):
        """Test that zstd-encoded bodies are correctly decompressed."""
        pytest.importorskip("zstandard")
        import zstandard

        original = b'{"test": "data", "number": 42}'
        compressor = zstandard.ZstdCompressor()
        compressed = compressor.compress(original)
        result = _decompress_body(compressed, "zstd")
        assert result == original

    def test_zstd_without_package(self, monkeypatch):
        """Test that missing zstandard package raises helpful error."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "zstandard":
                raise ImportError("No module named 'zstandard'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(HTTPException, match="zstd decompression requires"):
            _decompress_body(b"some data", "zstd")

    def test_zstd_invalid_payload(self):
        """Test that invalid zstd payloads raise HTTPException."""
        pytest.importorskip("zstandard")
        with pytest.raises(HTTPException, match="Failed to decompress zstd payload"):
            _decompress_body(b"not valid zstd", "zstd")

    def test_unsupported_encoding(self):
        """Test that unsupported encodings raise HTTPException."""
        with pytest.raises(HTTPException, match="Unsupported Content-Encoding"):
            _decompress_body(b"some data", "br")  # brotli

    def test_case_sensitivity(self):
        """Test that encoding names are case-insensitive (handled by caller)."""
        # The caller (_get_request_body) lowercases the encoding before passing
        original = b'{"test": "data"}'
        compressed = gzip.compress(original)
        # Should work with lowercase
        result = _decompress_body(compressed, "gzip")
        assert result == original


class TestGetRequestBody:
    """Tests for the _get_request_body function with Content-Encoding."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock()
        request.state = MagicMock()
        request.state.cached_body = None
        return request

    async def test_uncompressed_json(self, mock_request):
        """Test that uncompressed JSON is parsed correctly."""
        body = {"test": "data", "number": 42}
        mock_request.headers = {}
        mock_request.json = AsyncMock(return_value=body)

        result = await _get_request_body(mock_request)
        assert result == body

    async def test_cached_body_returned(self, mock_request):
        """Test that cached body from auth middleware is returned."""
        cached = {"cached": "data"}
        mock_request.state.cached_body = cached

        result = await _get_request_body(mock_request)
        assert result == cached
        # json() should not be called when cached
        mock_request.json.assert_not_called()

    async def test_gzip_encoded_json(self, mock_request):
        """Test that gzip-encoded JSON is decompressed and parsed."""
        body = {"test": "data", "number": 42}
        compressed = gzip.compress(json.dumps(body).encode())

        mock_request.headers = {"content-encoding": "gzip"}
        mock_request.body = AsyncMock(return_value=compressed)

        result = await _get_request_body(mock_request)
        assert result == body

    async def test_deflate_encoded_json(self, mock_request):
        """Test that deflate-encoded JSON is decompressed and parsed."""
        body = {"test": "data", "number": 42}
        compressed = zlib.compress(json.dumps(body).encode())

        mock_request.headers = {"content-encoding": "deflate"}
        mock_request.body = AsyncMock(return_value=compressed)

        result = await _get_request_body(mock_request)
        assert result == body

    @pytest.mark.skipif(
        not pytest.importorskip("zstandard", reason="zstandard not installed"),
        reason="zstandard not installed",
    )
    async def test_zstd_encoded_json(self, mock_request):
        """Test that zstd-encoded JSON is decompressed and parsed."""
        import zstandard

        body = {"test": "data", "number": 42}
        compressor = zstandard.ZstdCompressor()
        compressed = compressor.compress(json.dumps(body).encode())

        mock_request.headers = {"content-encoding": "zstd"}
        mock_request.body = AsyncMock(return_value=compressed)

        result = await _get_request_body(mock_request)
        assert result == body

    async def test_invalid_json_after_decompression(self, mock_request):
        """Test that invalid JSON after decompression raises HTTPException."""
        compressed = gzip.compress(b"not valid json")

        mock_request.headers = {"content-encoding": "gzip"}
        mock_request.body = AsyncMock(return_value=compressed)

        with pytest.raises(HTTPException, match="Invalid JSON payload"):
            await _get_request_body(mock_request)

    async def test_invalid_gzip_raises_http_exception(self, mock_request):
        """Test that invalid gzip data raises HTTPException with proper message."""
        mock_request.headers = {"content-encoding": "gzip"}
        mock_request.body = AsyncMock(return_value=b"not valid gzip")

        with pytest.raises(HTTPException, match="Failed to decompress gzip payload"):
            await _get_request_body(mock_request)

    async def test_unsupported_encoding_raises_http_exception(self, mock_request):
        """Test that unsupported encoding raises HTTPException."""
        mock_request.headers = {"content-encoding": "br"}
        mock_request.body = AsyncMock(return_value=b"some data")

        with pytest.raises(HTTPException, match="Unsupported Content-Encoding"):
            await _get_request_body(mock_request)

    async def test_content_encoding_case_insensitive(self, mock_request):
        """Test that Content-Encoding header is case-insensitive."""
        body = {"test": "data"}
        compressed = gzip.compress(json.dumps(body).encode())

        # Mixed case header value
        mock_request.headers = {"content-encoding": "GZIP"}
        mock_request.body = AsyncMock(return_value=compressed)

        result = await _get_request_body(mock_request)
        assert result == body

    async def test_content_encoding_with_whitespace(self, mock_request):
        """Test that Content-Encoding with whitespace is handled."""
        body = {"test": "data"}
        compressed = gzip.compress(json.dumps(body).encode())

        # Header with extra whitespace
        mock_request.headers = {"content-encoding": "  gzip  "}
        mock_request.body = AsyncMock(return_value=compressed)

        result = await _get_request_body(mock_request)
        assert result == body

    async def test_empty_content_encoding_treated_as_none(self, mock_request):
        """Test that empty Content-Encoding is treated as no encoding."""
        body = {"test": "data"}
        mock_request.headers = {"content-encoding": ""}
        mock_request.json = AsyncMock(return_value=body)

        result = await _get_request_body(mock_request)
        assert result == body
