import os
import shutil
import subprocess
from pathlib import Path

import click

import mlflow

mlflow_version = mlflow.version.VERSION


def build_docs(version):
    env = os.environ.copy()
    output_path = Path(f"_build/{version}")
    base_url = Path(f"/docs/{version}")
    build_path = Path("build")

    print(f"Building for `{base_url}`...")

    if output_path.exists():
        shutil.rmtree(output_path)

    if build_path.exists():
        shutil.rmtree(build_path)

    subprocess.check_call(["yarn", "build"], env={**env, "DOCS_BASE_URL": base_url})
    shutil.copytree(build_path, output_path)


@click.command()
def main():
    gtm_id = os.environ.get("GTM_ID")

    assert gtm_id, (
        "Google Tag Manager ID is missing, please ensure that the GTM_ID environment variable is set"
    )

    subprocess.check_call(["yarn", "install"])
    subprocess.check_call(["yarn", "build-api-docs"])
    subprocess.check_call(["yarn", "convert-notebooks"])

    output_path = Path("_build")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for v in [mlflow_version, "latest"]:
        build_docs(v)

    print("Finished building! Output can be found in the `_build` directory")


if __name__ == "__main__":
    main()
