"""
Install binary tools for MLflow development.
"""

# ruff: noqa: T201
import argparse
import gzip
import platform
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Type definitions
PlatformKey = tuple[
    Literal["linux", "darwin"],
    Literal["x86_64", "arm64"],
]
ExtractType = Literal["gzip", "tar", "binary"]


@dataclass
class Tool:
    name: str
    urls: dict[PlatformKey, str]  # platform -> URL mapping
    version_args: list[str] | None = None  # Custom version check args (default: ["--version"])

    def get_url(self, platform_key: PlatformKey) -> str | None:
        return self.urls.get(platform_key)

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
        urls={
            (
                "linux",
                "x86_64",
            ): "https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-linux-x86_64.gz",
            (
                "darwin",
                "arm64",
            ): "https://github.com/tamasfe/taplo/releases/download/0.9.3/taplo-darwin-aarch64.gz",
        },
    ),
    Tool(
        name="typos",
        urls={
            (
                "linux",
                "x86_64",
            ): "https://github.com/crate-ci/typos/releases/download/v1.28.0/typos-v1.28.0-x86_64-unknown-linux-musl.tar.gz",
            (
                "darwin",
                "arm64",
            ): "https://github.com/crate-ci/typos/releases/download/v1.28.0/typos-v1.28.0-aarch64-apple-darwin.tar.gz",
        },
    ),
    Tool(
        name="conftest",
        urls={
            (
                "linux",
                "x86_64",
            ): "https://github.com/open-policy-agent/conftest/releases/download/v0.63.0/conftest_0.63.0_Linux_x86_64.tar.gz",
            (
                "darwin",
                "arm64",
            ): "https://github.com/open-policy-agent/conftest/releases/download/v0.63.0/conftest_0.63.0_Darwin_arm64.tar.gz",
        },
    ),
    Tool(
        name="regal",
        urls={
            (
                "linux",
                "x86_64",
            ): "https://github.com/open-policy-agent/regal/releases/download/v0.36.1/regal_Linux_x86_64",
            (
                "darwin",
                "arm64",
            ): "https://github.com/open-policy-agent/regal/releases/download/v0.36.1/regal_Darwin_arm64",
        },
        version_args=["version"],
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


def extract_gzip_from_url(url: str, dest_dir: Path, binary_name: str) -> Path:
    print(f"Downloading from {url}")
    output_path = dest_dir / binary_name

    with urllib.request.urlopen(url) as response:
        with gzip.open(response, "rb") as gz:
            output_path.write_bytes(gz.read())

    return output_path


def extract_tar_from_url(url: str, dest_dir: Path, binary_name: str) -> Path:
    print(f"Downloading from {url}...")
    with (
        urllib.request.urlopen(url) as response,
        tarfile.open(fileobj=response, mode="r|*") as tar,
    ):
        # Find and extract only the binary file we need
        for member in tar:
            if member.isfile() and member.name.endswith(binary_name):
                # Extract the file content and write directly to destination
                tar.extract(member, dest_dir)
                return dest_dir / binary_name

    raise FileNotFoundError(f"Could not find {binary_name} in archive")


def download_binary_from_url(url: str, dest_dir: Path, binary_name: str) -> Path:
    print(f"Downloading from {url}...")
    output_path = dest_dir / binary_name

    with urllib.request.urlopen(url) as response:
        output_path.write_bytes(response.read())

    return output_path


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
        supported = [f"{os}-{arch}" for os, arch in tool.urls.keys()]
        raise RuntimeError(
            f"Current platform is not supported. Supported platforms: {', '.join(supported)}"
        )

    url = tool.get_url(platform_key)
    if url is None:
        os, arch = platform_key
        supported = [f"{os}-{arch}" for os, arch in tool.urls.keys()]
        raise RuntimeError(
            f"Platform {os}-{arch} not supported for {tool.name}. "
            f"Supported platforms: {', '.join(supported)}"
        )

    # Extract based on inferred type from URL
    extract_type = tool.get_extract_type(url)
    if extract_type == "gzip":
        binary_path = extract_gzip_from_url(url, dest_dir, tool.name)
    elif extract_type == "tar":
        binary_path = extract_tar_from_url(url, dest_dir, tool.name)
    elif extract_type == "binary":
        binary_path = download_binary_from_url(url, dest_dir, tool.name)
    else:
        raise ValueError(f"Unknown extract type: {extract_type}")

    # Make executable
    binary_path.chmod(0o755)

    # Verify installation by running version command
    version_cmd = [binary_path] + tool.get_version_args()
    subprocess.check_call(version_cmd, timeout=5)
    print(f"Successfully installed {tool.name} to {binary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install binary tools for MLflow development")
    parser.add_argument(
        "-f",
        "--force-reinstall",
        action="store_true",
        help="Force reinstall by removing existing tools",
    )
    args = parser.parse_args()

    if args.force_reinstall:
        print("Force reinstall: removing existing tools and reinstalling...")
    else:
        print("Installing all tools to bin/ directory...")

    dest_dir = Path(__file__).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    for tool in TOOLS:
        print(f"\nInstalling {tool.name}...")
        install_tool(tool, dest_dir, force=args.force_reinstall)

    print("\nAll tools installed successfully!")


if __name__ == "__main__":
    main()
