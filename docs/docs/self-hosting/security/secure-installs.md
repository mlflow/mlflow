# Secure Installs

Installing Python packages securely is a [best practice recommended by pip](https://pip.pypa.io/en/stable/topics/secure-installs/). This page describes how to apply these practices when installing MLflow and its dependencies.

## Hash-Checking Mode

pip's [hash-checking mode](https://pip.pypa.io/en/stable/topics/secure-installs/) verifies that every downloaded package matches a known SHA256 hash, protecting against tampering and man-in-the-middle attacks.

### Generating Hashed Requirements

Use [pip-compile](https://pip-tools.readthedocs.io/en/stable/) to resolve all dependencies and generate hashes:

```bash
pip install pip-tools
pip-compile --generate-hashes --output-file=requirements.txt requirements.in
```

Where `requirements.in` pins your direct dependencies:

```text
mlflow==3.10.1
```

The generated `requirements.txt` will include hashes for every package:

```text
mlflow==3.10.1 \
    --hash=sha256:abc123... \
    --hash=sha256:def456...
numpy==2.2.4 \
    --hash=sha256:789abc...
```

### Installing with Hash Verification

```bash
pip install --require-hashes -r requirements.txt
```

All packages (including transitive dependencies) must be pinned with `==` and have at least one hash. A partially-hashed requirements file is rejected by pip to prevent a malicious package from slipping through an unhashed entry.

### Disabling Source Distributions

To further reduce risk, use `--only-binary :all:` to prevent pip from running arbitrary build code (e.g., via `setup.py` or PEP 517 build backends) during installation:

```bash
pip install --require-hashes --only-binary :all: -r requirements.txt
```

## Filtering by Upload Time

Both pip and uv support filtering packages by their upload date, ensuring that dependency resolution only considers packages published before a known point in time. This protects against newly published malicious versions and enables reproducible builds.

:::tip Choosing a date
Use an absolute timestamp set to the last date you audited or verified your dependencies. This ensures reproducible installs. Relative durations like `"7 days"` are convenient for rolling policies but produce different results over time.
:::

### pip

pip's [`--uploaded-prior-to`](https://pip.pypa.io/en/latest/user_guide/#filtering-by-upload-time) option accepts an ISO 8601 datetime:

```bash
# Replace with a date you trust (e.g., the last date you verified your dependencies)
pip install --uploaded-prior-to="2026-03-01T00:00:00Z" mlflow
```

This only applies to packages from remote indexes (not local files or VCS requirements), and the index must provide upload-time metadata (PyPI does).

### uv

[uv](https://docs.astral.sh/uv/) provides the equivalent [`exclude-newer`](https://docs.astral.sh/uv/reference/settings/#exclude-newer) setting:

```bash
# Absolute timestamp - best for reproducible builds (replace with your own trusted date)
uv pip install --exclude-newer "2026-03-01T00:00:00Z" mlflow

# Relative duration - useful for rolling policies (e.g., always exclude packages newer than 7 days)
uv pip install --exclude-newer "7 days" mlflow
```

For one-off tool invocations with `uvx`:

```bash
# Absolute timestamp - best for reproducible builds (replace with your own trusted date)
uvx --exclude-newer "2026-03-01T00:00:00Z" mlflow server

# Relative duration - useful for rolling policies (e.g., always exclude packages newer than 7 days)
uvx --exclude-newer "7 days" mlflow server
```
