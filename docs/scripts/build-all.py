import os
import shutil
import subprocess
from pathlib import Path

import click

import mlflow

mlflow_version = mlflow.version.VERSION


def build_docs(package_manager, version):
    env = os.environ.copy()
    output_path = Path(f"_build/{version}")
    base_url = Path(f"/docs/{version}")
    build_path = Path("build")

    print(f"Building for `{base_url}`...")

    if output_path.exists():
        shutil.rmtree(output_path)

    if build_path.exists():
        shutil.rmtree(build_path)

    subprocess.check_call(package_manager + ["build"], env={**env, "DOCS_BASE_URL": base_url})
    shutil.copytree(build_path, output_path)


@click.command()
@click.option(
    "--use-npm",
    "use_npm",
    is_flag=True,
    default=False,
    help="Whether or not to use NPM as a package manager (in case yarn in unavailable)",
)
def main(use_npm):
    gtm_id = os.environ.get("GTM_ID")

    assert gtm_id, (
        "Google Tag Manager ID is missing, please ensure that the GTM_ID environment variable is set"
    )

    package_manager = ["npm", "run"] if use_npm else ["yarn"]

    subprocess.check_call(package_manager + ["build-api-docs"])
    subprocess.check_call(package_manager + ["convert-notebooks"])

    output_path = Path("_build")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for v in [mlflow_version, "latest"]:
        build_docs(package_manager, v)

    final_path = Path("build")
    if final_path.exists():
        shutil.rmtree(final_path)

    shutil.move(output_path, final_path)

    print("Finished building! Output can be found in the `build` directory")


if __name__ == "__main__":
    main()
