# Triage runner image

The `triage` job in `triage.yml` uses the GitHub-hosted `ubuntu-triage` runner.
Each job starts on a fresh VM with the `mlflow-triage` custom image. Other triage
jobs continue to use standard GitHub-hosted runners.

`triage-runner-image.yml` prepares the system packages, Claude Code, agent-browser,
and sandbox runtime. It runs `uv sync --locked` and `yarn install --immutable` to
populate the Python and frontend package caches. The consumer checks out the
current commit and uses its lockfiles; the cached packages do not replace source
code or lockfile validation. Cache writes stay inside each disposable VM.

The image is rebuilt weekly and when its workflow, tool setup, or dependency
lockfiles change on `master`. To refresh it manually:

```bash
gh workflow run triage-runner-image.yml --repo mlflow/mlflow --ref master
```

After the image becomes available, verify the tools, sandbox, browser, and package
caches without downloading Python or frontend packages:

```bash
gh workflow run triage-runner-smoke-test.yml --repo mlflow/mlflow --ref master
```

## Organization configuration

1. Create a **Triage Image Builders** runner group, select only `mlflow/mlflow`,
   allow public repositories, and restrict workflow access to
   `mlflow/mlflow/.github/workflows/triage-runner-image.yml@refs/heads/master`.
   GitHub requires that workflow to exist at the selected ref. When bootstrapping
   from an upstream branch, allow only that workflow at the branch ref, then
   replace it with `master` after merging.
2. Create **ubuntu-triage-image-builder** in that group: Linux x64, Ubuntu 24.04,
   image generation enabled, 4 CPUs, 16 GB RAM, 150 GB SSD, maximum concurrency 1.
3. Run the image workflow and wait for `mlflow-triage` to become available in
   organization Settings > Actions > Custom images. Completing the build job
   does not mean snapshot provisioning has finished.
4. Create **ubuntu-triage** using the `mlflow-triage` image, with image version
   **Latest**, the same platform and size, and a runner group that permits
   `mlflow/mlflow`. Do not enable image generation on this consumer runner.

Keep image builders separate from consumer runners and do not grant builders to
other workflows or pull request refs. Image building needs only `contents: read`;
it does not use Databricks credentials or run the issue agent.

Larger runners are billed even for public repositories, and custom image storage
is billed separately. See [GitHub's runner pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing).
