import asyncio
import json
import ssl
import subprocess
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from skills.github import GitHubClient


def test_client_uses_environment_proxy_for_api_and_streamed_redirect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cert_path = tmp_path / "proxy-cert.pem"
    key_path = tmp_path / "proxy-key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "1",
            "-subj",
            "/CN=api.github.com",
            "-addext",
            ("subjectAltName=DNS:api.github.com,DNS:productionresultssa12.blob.core.windows.net"),
            "-addext",
            "basicConstraints=critical,CA:TRUE",
        ],
        check=True,
        capture_output=True,
    )

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    original_client_session = aiohttp.ClientSession

    def create_client_session(*args: Any, **kwargs: Any) -> aiohttp.ClientSession:
        assert kwargs["trust_env"] is True
        kwargs["connector"] = aiohttp.TCPConnector(ssl=False)
        return original_client_session(*args, **kwargs)

    monkeypatch.setattr(aiohttp, "ClientSession", create_client_session)

    async def run_test() -> None:
        requests: list[tuple[str, str, str | None]] = []
        proxy_destinations: list[str] = []
        token = "test-token"
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.load_cert_chain(cert_path, key_path)

        async def handle_request(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            request_line = await reader.readline()
            headers: dict[str, str] = {}
            while line := await reader.readline():
                if line == b"\r\n":
                    break
                name, value = line.decode().split(":", 1)
                headers[name.lower()] = value.strip()

            method, path, _ = request_line.decode().split()
            host = headers["host"]
            requests.append((host, path, headers.get("authorization")))
            if path == "/repos/owner/repo/actions/jobs/1":
                body = json.dumps({"job": "ok"}).encode()
                response = (
                    b"HTTP/1.1 200 OK\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"Content-Type: application/json\r\n"
                    + b"Connection: close\r\n\r\n"
                    + body
                )
            elif host == "api.github.com" and method == "GET" and path.endswith("/logs"):
                response = (
                    b"HTTP/1.1 302 Found\r\n"
                    b"Location: https://productionresultssa12.blob.core.windows.net/logs\r\n"
                    b"Content-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
            else:
                body = b"streamed-log-content"
                response = (
                    b"HTTP/1.1 200 OK\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                    + body
                )
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        backend = await asyncio.start_server(handle_request, "127.0.0.1", 0, ssl=tls_context)
        backend_port = backend.sockets[0].getsockname()[1]

        async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while chunk := await reader.read(64 * 1024):
                    writer.write(chunk)
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        async def handle_proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            connect_line = await reader.readline()
            destination = connect_line.decode().split()[1]
            proxy_destinations.append(destination)
            while await reader.readline() != b"\r\n":
                pass
            backend_reader, backend_writer = await asyncio.open_connection(
                "127.0.0.1", backend_port
            )
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await asyncio.gather(relay(reader, backend_writer), relay(backend_reader, writer))

        proxy = await asyncio.start_server(handle_proxy, "127.0.0.1", 0)
        proxy_port = proxy.sockets[0].getsockname()[1]
        monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{proxy_port}")

        try:
            async with GitHubClient(token) as client:
                assert await client._get_json("/repos/owner/repo/actions/jobs/1") == {"job": "ok"}
                async with await client.get_raw(
                    "/repos/owner/repo/actions/jobs/1/logs"
                ) as response:
                    response.raise_for_status()
                    content = b"".join([chunk async for chunk in response.content.iter_chunked(7)])
                assert content == b"streamed-log-content"
        finally:
            proxy.close()
            backend.close()
            await proxy.wait_closed()
            await backend.wait_closed()

        assert proxy_destinations == [
            "api.github.com:443",
            "api.github.com:443",
            "productionresultssa12.blob.core.windows.net:443",
        ]
        assert [(host, path) for host, path, _ in requests] == [
            ("api.github.com", "/repos/owner/repo/actions/jobs/1"),
            ("api.github.com", "/repos/owner/repo/actions/jobs/1/logs"),
            ("productionresultssa12.blob.core.windows.net", "/logs"),
        ]
        assert requests[0][2] is not None
        assert requests[1][2] is not None
        assert requests[2][2] is None

    asyncio.run(run_test())
