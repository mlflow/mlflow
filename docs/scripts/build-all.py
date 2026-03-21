import os
import shutil
import subprocess
from pathlib import Path

import click

import mlflow

mlflow_version = mlflow.version.VERSION


def build_docs(package_manager, version):
    env = os.environ.copy()

    # ensure it ends with a "/"
    base_url = env.get("DOCS_BASE_URL", "/docs/").rstrip("/") + "/"
    api_reference_prefix = (
        env.get("API_REFERENCE_PREFIX", "https://mlflow.org/docs/").rstrip("/") + "/"
    )

    output_path = Path(f"_build/{version}")
    versioned_url = Path(f"{base_url}{version}")
    build_path = Path("build")

    print(f"Building for `{versioned_url}`...")

    if output_path.exists():
        shutil.rmtree(output_path)

    if build_path.exists():
        shutil.rmtree(build_path)

    subprocess.check_call(
        package_manager + ["build"],
        env={
            **env,
            "DOCS_BASE_URL": str(versioned_url),
            "API_REFERENCE_PREFIX": f"{api_reference_prefix}{version}",
        },
    )
    shutil.copytree(build_path, output_path)


@click.command()
@click.option(
    "--use-npm",
    "use_npm",
    is_flag=True,
    default=False,
    help="Whether or not to use NPM as a package manager (in case yarn in unavailable)",
)
@click.option(
    "--no-r",
    "no_r",
    is_flag=True,
    default=False,
    help="Whether or not to skip building R documentation.",
)
def main(use_npm, no_r):
    gtm_id = os.environ.get("GTM_ID")

    assert gtm_id, (
        "Google Tag Manager ID is missing, please ensure that the GTM_ID environment variable is set"
    )

    package_manager = ["npm", "run"] if use_npm else ["yarn"]
    build_command = ["build-api-docs:no-r"] if no_r else ["build-api-docs"]

    subprocess.check_call(package_manager + build_command)
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
