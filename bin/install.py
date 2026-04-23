"""
Install binary tools for MLflow development.
"""

# ruff: noqa: T201
import argparse
import gzip
import hashlib
import http.client
import json
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError

INSTALLED_VERSIONS_FILE = ".installed_versions.json"

# Type definitions
PlatformKey = tuple[
    Literal["linux", "darwin"],
    Literal["x86_64", "arm64"],
]
ExtractType = Literal["gzip", "tar", "binary"]


@dataclass
class Tool:
    name: str
    version: str
    assets: dict[PlatformKey, tuple[str, str]]  # platform -> (url, sha256)
    version_args: list[str] | None = None  # Custom version check args (default: ["--version"])

    def get_asset(self, platform_key: PlatformKey) -> tuple[str, str] | None:
        return self.assets.get(platform_key)

    def get_version_args(self) -> list[str]:
        """Get version check arguments, defaulting to --version."""
        return self.version_args or ["--version"]

    def get_extract_type(self, url: str) -> ExtractType:
        """Infer extract type from URL file extension."""
        if url.endswith(".gz") and not url.endswith(".tar.gz"):
            return "gzip"
        elif url.endswith((".tar.gz", ".tgz")):
            return "tar"
        elif url.endswith(".exe") or ("/" in url and not url.split("/")[-1].count(".")):
            # Windows executables or files without extensions (plain binaries)
            return "binary"
        else:
            # Default to tar for unknown extensions
            return "tar"


# Tool configurations
TOOLS = [
    Tool(
        name="taplo",
        version="0.9.3",
        assets={
            ("linux", "x86_64"): (
                "https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-linux-x86_64.gz",
                "889efcfa067b179fda488427d3b13ce2d679537da8b9ed8138ba415db7da2a5e",
            ),
            ("darwin", "arm64"): (
                "https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-darwin-aarch64.gz",
                "39b84d62d6a47855b2c64148cde9c9ca5721bf422b8c9fe9c92776860badde5f",
            ),
        },
    ),
    Tool(
        name="typos",
        version="1.39.2",
        assets={
            ("linux", "x86_64"): (
                "https://github.com/crate-ci/typos/releases/download/v1.39.2/typos-v1.39.2-x86_64-unknown-linux-musl.tar.gz",
                "4acfb2123a9a295d34a411ad90af23717d06914c58023ab1a12b6605f0ce3e3c",
            ),
            ("darwin", "arm64"): (
                "https://github.com/crate-ci/typos/releases/download/v1.39.2/typos-v1.39.2-aarch64-apple-darwin.tar.gz",
                "1dac53624939bf7b638df8cd168af46532f4fbad2b512c8b092cdf1487b94612",
            ),
        },
    ),
    Tool(
        name="conftest",
        version="0.63.0",
        assets={
            ("linux", "x86_64"): (
                "https://github.com/open-policy-agent/conftest/releases/download/v0.63.0/conftest_0.63.0_Linux_x86_64.tar.gz",
                "59b354bedf0d761fb562404a8af3015a48415636382f975a2037ca81c0c6202f",
            ),
            ("darwin", "arm64"): (
                "https://github.com/open-policy-agent/conftest/releases/download/v0.63.0/conftest_0.63.0_Darwin_arm64.tar.gz",
                "026378585ed42609f23996663c2feea9535bc19dc3909a99dabe776b7708b85c",
            ),
        },
    ),
    Tool(
        name="regal",
        version="0.36.1",
        assets={
            ("linux", "x86_64"): (
                "https://github.com/open-policy-agent/regal/releases/download/v0.36.1/regal_Linux_x86_64",
                "75509b89de9d2fa12ac30157cc7269e7abc61e8c4c407a29ce897b681a78f8a4",
            ),
            ("darwin", "arm64"): (
                "https://github.com/open-policy-agent/regal/releases/download/v0.36.1/regal_Darwin_arm64",
                "66d1578885bf8fb7a4bd7b435a74acf8205af7fc49d6db84b6df0cddba9d7591",
            ),
        },
        version_args=["version"],
    ),
    Tool(
        name="buf",
        version="1.59.0",
        assets={
            ("linux", "x86_64"): (
                "https://github.com/bufbuild/buf/releases/download/v1.59.0/buf-Linux-x86_64",
                "d7462609e3814629c642ac10f0e7e27ec7e8e21d1dd75742f4434c31619e986b",
            ),
            ("darwin", "arm64"): (
                "https://github.com/bufbuild/buf/releases/download/v1.59.0/buf-Darwin-arm64",
                "71f060640b9f1a3fce43db31eb8e8faf714a3bfbbcb70617946bdeba3aadf56b",
            ),
        },
    ),
    Tool(
        name="rg",
        version="14.1.1",
        assets={
            ("linux", "x86_64"): (
                "https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz",
                "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e",
            ),
            ("darwin", "arm64"): (
                "https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-aarch64-apple-darwin.tar.gz",
                "24ad76777745fbff131c8fbc466742b011f925bfa4fffa2ded6def23b5b937be",
            ),
        },
    ),
]


def get_platform_key() -> PlatformKey | None:
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize machine architecture
    if machine in ["x86_64", "amd64"]:
        machine = "x86_64"
    elif machine in ["aarch64", "arm64"]:
        machine = "arm64"

    # Return if it's a supported platform combination
    if system == "linux" and machine == "x86_64":
        return ("linux", "x86_64")
    elif system == "darwin" and machine == "arm64":
        return ("darwin", "arm64")

    return None


def urlopen_with_retry(
    url: str, max_retries: int = 7, base_delay: float = 1.0
) -> http.client.HTTPResponse:
    """Open a URL with retry logic for transient HTTP errors (e.g., 503)."""
    for attempt in range(max_retries):
        try:
            return urllib.request.urlopen(url)
        except HTTPError as e:
            if e.code in (502, 503, 504) and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"  HTTP {e.code}, retrying in {delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
        except (http.client.RemoteDisconnected, ConnectionResetError, URLError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"  {e}, retrying in {delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def download_and_verify(url: str, dest: Path, expected_sha256: str) -> None:
    print(f"Downloading from {url}...")
    sha256 = hashlib.sha256()
    with urlopen_with_retry(url) as response, dest.open("wb") as f:
        while chunk := response.read(1024 * 1024):
            sha256.update(chunk)
            f.write(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {url}\n  expected: {expected_sha256}\n  actual:   {actual}"
        )


def extract_gzip(download_path: Path, dest: Path) -> None:
    with gzip.open(download_path, "rb") as gz:
        dest.write_bytes(gz.read())


def extract_tar(download_path: Path, dest: Path, binary_name: str) -> None:
    with tarfile.open(download_path, mode="r:*") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(binary_name):
                f = tar.extractfile(member)
                if f is not None:
                    dest.write_bytes(f.read())
                    return
    raise FileNotFoundError(f"Could not find {binary_name} in archive")


def install_tool(tool: Tool, dest_dir: Path, force: bool = False) -> None:
    # Check if tool already exists
    binary_path = dest_dir / tool.name
    if binary_path.exists():
        if not force:
            print(f"  ✓ {tool.name} already installed")
            return
        else:
            print(f"  Removing existing {tool.name}...")
            binary_path.unlink()

    platform_key = get_platform_key()

    if platform_key is None:
        supported = [f"{os}-{arch}" for os, arch in tool.assets.keys()]
        raise RuntimeError(
            f"Current platform is not supported. Supported platforms: {', '.join(supported)}"
        )

    asset = tool.get_asset(platform_key)
    if asset is None:
        os, arch = platform_key
        supported = [f"{os}-{arch}" for os, arch in tool.assets.keys()]
        raise RuntimeError(
            f"Platform {os}-{arch} not supported for {tool.name}. "
            f"Supported platforms: {', '.join(supported)}"
        )
    url, expected_sha256 = asset

    binary_path = dest_dir / tool.name
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_path = Path(tmp_dir) / "download"
        download_and_verify(url, download_path, expected_sha256)

        extract_type = tool.get_extract_type(url)
        if extract_type == "gzip":
            extract_gzip(download_path, binary_path)
        elif extract_type == "tar":
            extract_tar(download_path, binary_path, tool.name)
        elif extract_type == "binary":
            shutil.move(download_path, binary_path)
        else:
            raise ValueError(f"Unknown extract type: {extract_type}")

    # Make executable
    binary_path.chmod(0o755)

    # Verify installation by running version command
    version_cmd = [binary_path] + tool.get_version_args()
    subprocess.check_call(version_cmd, timeout=5)
    print(f"Successfully installed {tool.name} to {binary_path}")


def load_installed_versions(dest_dir: Path) -> dict[str, str]:
    f = dest_dir / INSTALLED_VERSIONS_FILE
    if f.exists():
        return json.loads(f.read_text())
    return {}


def save_installed_versions(dest_dir: Path, versions: dict[str, str]) -> None:
    f = dest_dir / INSTALLED_VERSIONS_FILE
    f.write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n")


def main() -> None:
    all_tool_names = [t.name for t in TOOLS]
    parser = argparse.ArgumentParser(description="Install binary tools for MLflow development")
    parser.add_argument(
        "-f",
        "--force-reinstall",
        action="store_true",
        help="Force reinstall by removing existing tools",
    )
    parser.add_argument(
        "tools",
        nargs="*",
        metavar="TOOL",
        help=f"Tools to install (default: all). Available: {', '.join(all_tool_names)}",
    )
    args = parser.parse_args()

    # Filter tools if specific ones requested
    if args.tools:
        if invalid_tools := set(args.tools) - set(all_tool_names):
            parser.error(
                f"Unknown tools: {', '.join(sorted(invalid_tools))}. "
                f"Available: {', '.join(all_tool_names)}"
            )
        tools_to_install = [t for t in TOOLS if t.name in args.tools]
    else:
        tools_to_install = TOOLS

    dest_dir = Path(__file__).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    installed_versions = load_installed_versions(dest_dir)
    outdated_tools = sorted(
        t.name for t in tools_to_install if installed_versions.get(t.name) != t.version
    )
    force_all = args.force_reinstall

    if force_all:
        print("Force reinstall: removing existing tools and reinstalling...")
    elif outdated_tools:
        print(f"Version changes detected for: {', '.join(outdated_tools)}")
    else:
        print("Installing tools to bin/ directory...")

    for tool in tools_to_install:
        # Force reinstall if globally forced or if this tool's version changed
        force = force_all or tool.name in outdated_tools
        print(f"\nInstalling {tool.name}...")
        install_tool(tool, dest_dir, force=force)
        installed_versions[tool.name] = tool.version

    save_installed_versions(dest_dir, installed_versions)
    print("\nDone!")


if __name__ == "__main__":
    main()
