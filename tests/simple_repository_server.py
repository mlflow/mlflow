"""PEP 700-compliant Simple Repository API server for serving wheels in tests.

This replaces the plain ``http.server`` approach so that uv's ``exclude-newer``
can filter packages by upload time when resolving the local dev wheel.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from typing_extensions import Self


def _make_handler(wheel_dir: Path) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?")[0].split("#")[0].rstrip("/")
            if path in ("", "/simple"):
                self._serve_index()
            elif path == "/simple/mlflow":
                self._serve_project()
            else:
                super().do_GET()

        def _wants_json(self) -> bool:
            accept = self.headers.get("Accept", "")
            return "application/vnd.pypi.simple.v1+json" in accept

        def _serve_index(self) -> None:
            if self._wants_json():
                body = json.dumps({
                    "meta": {"api-version": "1.1"},
                    "projects": [{"name": "mlflow"}],
                }).encode()
                content_type = "application/vnd.pypi.simple.v1+json"
            else:
                body = b"<html><body><a href='/simple/mlflow/'>mlflow</a></body></html>"
                content_type = "text/html"
            self._respond(200, content_type, body)

        def _serve_project(self) -> None:
            files = []
            versions: set[str] = set()
            for f in sorted(wheel_dir.iterdir()):
                if f.suffix != ".whl":
                    continue
                sha256 = hashlib.sha256(f.read_bytes()).hexdigest()
                version = f.stem.split("-")[1]
                versions.add(version)
                files.append({
                    "filename": f.name,
                    "url": f"/mlflow/{f.name}#sha256={sha256}",
                    "hashes": {"sha256": sha256},
                    "size": f.stat().st_size,
                    # A date safely before any exclude-newer cutoff so the dev
                    # wheel is always resolvable.
                    "upload-time": "2020-01-01T00:00:00Z",
                })

            if self._wants_json():
                body = json.dumps({
                    "meta": {"api-version": "1.1"},
                    "name": "mlflow",
                    "versions": sorted(versions),
                    "files": files,
                }).encode()
                content_type = "application/vnd.pypi.simple.v1+json"
            else:
                links = "".join(
                    f'<a href="{f["url"]}" data-dist-info-metadata="false">{f["filename"]}</a>'
                    for f in files
                )
                body = f"<html><body>{links}</body></html>".encode()
                content_type = "text/html"
            self._respond(200, content_type, body)

        def _respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def translate_path(self, path: str) -> str:
            path = path.split("?")[0].split("#")[0]
            return str(wheel_dir.parent / path.lstrip("/"))

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


class SimpleRepositoryServer:
    def __init__(self, wheel_dir: Path, port: int) -> None:
        handler = _make_handler(wheel_dir)
        self._server = HTTPServer(("", port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="simple-repository-server"
        )

    @property
    def url(self) -> str:
        _, port = self._server.server_address
        return f"http://localhost:{port}/simple"

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
        self._thread.join()
