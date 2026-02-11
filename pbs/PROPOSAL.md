# Proposal: Replace pyenv with python-build-standalone in MLflow

## Executive Summary

This proposal recommends replacing pyenv with [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (PBS) for Python version management in MLflow's virtualenv environment manager. PBS provides pre-built Python binaries that eliminate compilation time, reduce dependencies, and significantly improve setup performance in CI/CD pipelines and Docker image builds.

## Background

### Current State: pyenv in MLflow

MLflow currently uses pyenv for Python version management when `env_manager="virtualenv"` is specified. Key usage areas include:

| Component             | File                            | Purpose                                     |
| --------------------- | ------------------------------- | ------------------------------------------- |
| Core virtualenv logic | `mlflow/utils/virtualenv.py`    | Install Python versions, create virtualenvs |
| Docker image building | `mlflow/models/docker_utils.py` | Setup script for Docker images              |
| CI/CD pipelines       | `.github/actions/setup-pyenv/`  | GitHub Actions setup                        |
| Development setup     | `dev/dev-env-setup.sh`          | Local development environment               |
| Test Dockerfiles      | `tests/resources/dockerfile/`   | Test environment images                     |

### Problems with pyenv

1. **Slow compilation**: pyenv compiles Python from source, taking 5-15 minutes depending on the system
2. **Build dependencies**: Requires extensive build tools (gcc, make, libssl-dev, etc.)
3. **Platform inconsistency**: Different installation methods for Linux (git clone), macOS (Homebrew), Windows (pyenv-win)
4. **Fragile builds**: Compilation can fail due to missing dependencies or version conflicts
5. **Large Docker images**: Build tools increase image size significantly
6. **CI time consumption**: Significant portion of CI time spent on Python compilation

## Proposed Solution: python-build-standalone

### What is PBS?

[python-build-standalone](https://github.com/astral-sh/python-build-standalone) is a project maintained by Astral (creators of uv, ruff) that provides pre-built, portable Python distributions for multiple platforms.

### Key Benefits

| Aspect                | pyenv                        | PBS                                |
| --------------------- | ---------------------------- | ---------------------------------- |
| **Installation time** | 5-15 minutes (compile)       | 10-30 seconds (download + extract) |
| **Dependencies**      | gcc, make, libssl-dev, etc.  | None (just curl/wget + tar)        |
| **Reliability**       | Can fail due to build issues | Consistent pre-built binaries      |
| **Docker image size** | +500MB build tools           | Minimal overhead                   |
| **Cross-platform**    | Different tools per OS       | Unified approach                   |
| **Maintenance**       | Complex scripts              | Simple download logic              |

### Supported Platforms

PBS provides pre-built binaries for:

- Linux: x86_64, aarch64, i686
- macOS: x86_64, aarch64 (Apple Silicon)
- Windows: x86_64, i686

### Available Python Versions

- CPython 3.8 through 3.14 (latest)
- PyPy support
- Free-threaded Python 3.13+ variants

## Implementation Plan

### Phase 1: Core Integration

Replace pyenv calls in `mlflow/utils/virtualenv.py` with direct PBS downloads:

```python
import platform
import tarfile
import urllib.request
from pathlib import Path

# PBS release configuration
PBS_RELEASE = "20260113"
PBS_BASE_URL = "https://github.com/astral-sh/python-build-standalone/releases/download"


def _get_pbs_download_url(version: str) -> str:
    """Construct PBS download URL for the current platform."""
    arch = platform.machine()
    system = platform.system().lower()

    # Map platform names
    if system == "darwin":
        system = "apple-darwin"
    elif system == "linux":
        system = "unknown-linux-gnu"
    elif system == "windows":
        system = "pc-windows-msvc-shared"

    filename = f"cpython-{version}+{PBS_RELEASE}-{arch}-{system}-install_only.tar.gz"
    return f"{PBS_BASE_URL}/{PBS_RELEASE}/{filename}"


def _install_python_pbs(version: str, install_root: str) -> str:
    """Download and install Python from PBS."""
    install_dir = Path(install_root) / f"cpython-{version}"

    if install_dir.exists():
        return str(install_dir / "bin" / "python")

    url = _get_pbs_download_url(version)
    tarball_path = Path(install_root) / "python.tar.gz"

    # Download
    urllib.request.urlretrieve(url, tarball_path)

    # Extract
    install_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path) as tf:
        tf.extractall(install_dir)

    tarball_path.unlink()
    return str(install_dir / "python" / "bin" / "python")
```

### Phase 2: Docker Integration

Update `mlflow/models/docker_utils.py`:

```python
# Before (pyenv)
SETUP_PYENV_AND_VIRTUALENV = """
# Install pyenv dependencies
apt-get update && apt-get install -y git curl make build-essential \\
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev

# Install pyenv
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python (compiles from source - slow!)
pyenv install {python_version}
pyenv global {python_version}
"""

# After (PBS - direct download)
SETUP_PBS_AND_VIRTUALENV = """
# Minimal dependencies
apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download pre-built Python from PBS (fast!)
ARCH=$(uname -m)
curl -fsSL -o /tmp/python.tar.gz \\
    "https://github.com/astral-sh/python-build-standalone/releases/download/{pbs_release}/cpython-{python_version}+{pbs_release}-${{ARCH}}-unknown-linux-gnu-install_only.tar.gz"
mkdir -p /opt/python
tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1
rm /tmp/python.tar.gz
export PATH="/opt/python/bin:$PATH"
"""
```

## Benchmark Results

See [benchmark/](./benchmark/) directory for full benchmark scripts and results.

### Python Installation Time Comparison

| Approach                    | Python Install Time | Image Size   |
| --------------------------- | ------------------- | ------------ |
| PBS (download + extract)    | **27.4s**           | **535.8 MB** |
| pyenv (compile from source) | 160.5s\*            | 1708.4 MB    |

\*pyenv total includes: build dependencies (123.2s) + pyenv clone (2.6s) + Python compilation (34.7s)

### Key Metrics

- **Speed improvement**: PBS is **5.9x faster** for Python setup
- **Image size reduction**: **1172.6 MB smaller** (68% reduction)
- **Dependency reduction**: PBS requires only `curl` and `ca-certificates` vs 264 packages for pyenv

### Notes on Benchmark

- Benchmark run on Apple Silicon (M-series) with fast compilation
- On typical CI machines, pyenv compilation takes 5-15 minutes
- PBS download time depends mainly on network speed (~27MB tarball)
- Expected speedup in CI: **10-30x faster**

## Migration Strategy

### Backward Compatibility

1. **Deprecation period**: Support both pyenv and PBS for 2-3 releases
2. **Configuration option**: Add `MLFLOW_PYTHON_INSTALLER` env var (`pyenv` | `pbs`)
3. **Default transition**:
   - v2.x: Default to pyenv, PBS opt-in
   - v3.x: Default to PBS, pyenv opt-in
   - v4.x: Remove pyenv support

### Fallback Mechanism

```python
def _install_python(version, install_root=None):
    installer = os.environ.get("MLFLOW_PYTHON_INSTALLER", "pbs")

    if installer == "pbs":
        return _install_python_pbs(version, install_root)
    elif installer == "pyenv":
        return _install_python_pyenv(version, install_root)
    else:
        raise ValueError(f"Unknown installer: {installer}")
```

## Risk Assessment

### Low Risk

- PBS is maintained by Astral (same team behind uv, which MLflow already supports)
- Pre-built binaries are widely used (uv, rye, mise all use PBS)
- No compilation means fewer failure modes
- SHA256 checksums available for all binaries

### Medium Risk

#### 1. Network/Firewall Restrictions

**Risk**: Corporate firewalls or air-gapped environments may block GitHub releases.

**Mitigations**:

- Support configurable mirror URLs via `MLFLOW_PBS_MIRROR_URL` environment variable
- Provide documentation for setting up internal mirrors
- Fall back to pyenv if PBS download fails (during transition period)

```python
PBS_BASE_URL = os.environ.get(
    "MLFLOW_PBS_MIRROR_URL",
    "https://github.com/astral-sh/python-build-standalone/releases/download",
)
```

#### 2. GitHub API Rate Limits

**Risk**: GitHub has rate limits (60 requests/hour unauthenticated, 5000/hour authenticated).

**Mitigations**:

- PBS downloads are from GitHub Releases (not API), which have higher limits
- Support `GITHUB_TOKEN` environment variable for authenticated requests
- Cache downloaded binaries locally to avoid repeated downloads
- Use `curl --retry` for transient failures

#### 3. Version Availability Lag

**Risk**: New Python releases may not be immediately available in PBS.

**Mitigations**:

- PBS typically releases within 1-2 weeks of CPython releases
- Maintain a version mapping table for supported versions
- Fall back to pyenv for unsupported versions
- Document supported version ranges

#### 4. Binary Compatibility

**Risk**: PBS binaries may have compatibility issues with certain Linux distributions.

**Mitigations**:

- PBS provides multiple variants (glibc versions, musl, etc.)
- Test on target distributions before deployment
- Document supported distributions

### Risk Summary Table

| Risk                     | Likelihood | Impact | Mitigation                             |
| ------------------------ | ---------- | ------ | -------------------------------------- |
| Firewall blocking GitHub | Medium     | High   | Mirror URL support, fallback to pyenv  |
| GitHub rate limits       | Low        | Medium | Use releases (not API), caching, retry |
| Version availability     | Low        | Low    | Version mapping, pyenv fallback        |
| Binary compatibility     | Low        | Medium | Multiple variants, testing             |
| PBS project discontinued | Very Low   | High   | Astral backing, widespread adoption    |

## Files to Modify

| File                            | Changes                                              |
| ------------------------------- | ---------------------------------------------------- |
| `mlflow/utils/virtualenv.py`    | Add PBS download functions, update `_install_python` |
| `mlflow/models/docker_utils.py` | Update Docker setup script                           |
| `.github/actions/setup-pyenv/`  | Replace with PBS-based action or rename              |
| `dev/dev-env-setup.sh`          | Update development setup                             |
| `tests/resources/dockerfile/*`  | Update test Dockerfiles                              |

## Conclusion

Replacing pyenv with python-build-standalone will:

1. **Reduce build time** by 1.7x (or 10-20x in CI environments)
2. **Shrink Docker images** by ~1.2 GB (68% smaller)
3. **Improve reliability** with pre-built binaries
4. **Simplify maintenance** with unified cross-platform approach
5. **Align with ecosystem** (uv, rye, mise already use PBS)

The migration can be done incrementally with full backward compatibility, making this a low-risk, high-reward improvement.

## References

- [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
- [PBS Release Notes](https://github.com/astral-sh/python-build-standalone/releases)
- [uv Python management](https://docs.astral.sh/uv/concepts/python-versions/)
- [MLflow virtualenv implementation](https://github.com/mlflow/mlflow/blob/master/mlflow/utils/virtualenv.py)
