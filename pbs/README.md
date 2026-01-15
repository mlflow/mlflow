# PBS vs pyenv Benchmark

Comparing Python installation approaches:

- **PBS**: Downloads pre-built Python from [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
- **pyenv**: Compiles Python from source

## Results

| Approach | Python Install Time | Image Size   |
| -------- | ------------------- | ------------ |
| PBS      | **27.4s**           | **535.8 MB** |
| pyenv    | 160.5s              | 1708.4 MB    |

**PBS is 5.9x faster** and produces **68% smaller** images.

## Run Locally

```bash
# PBS benchmark (fast)
docker build -f Dockerfile.pbs -t benchmark-pbs .
docker run --rm benchmark-pbs

# pyenv benchmark (slow)
docker build -f Dockerfile.pyenv -t benchmark-pyenv .
docker run --rm benchmark-pyenv
```

## GitHub Actions

The benchmark runs automatically via `.github/workflows/pbs-benchmark.yml`.
