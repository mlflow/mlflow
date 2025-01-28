import os
import shutil
import subprocess

import click

import mlflow

mlflow_version = mlflow.version.VERSION


def build_docs(version):
    env = os.environ.copy()
    output_path = f"_build/{version}"
    base_url = f"/docs/{version}"

    print(f"Building for `{base_url}`...")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if os.path.exists("build"):
        shutil.rmtree("build")

    subprocess.run(["yarn", "build"], env={**env, "DOCS_BASE_URL": base_url})
    shutil.copytree("build", output_path)


@click.command()
def main():
    env = os.environ.copy()
    gtm_id = env.get("GTM_ID")

    assert (
        gtm_id
    ), "Google Tag Manager ID is missing, please ensure that the GTM_ID environment variable is set"

    subprocess.run(["yarn", "install"])
    subprocess.run(["yarn", "build-api-docs"], env=env)
    subprocess.run(["yarn", "convert-notebooks"], env=env)

    if os.path.exists("_build"):
        shutil.rmtree("_build")
    os.makedirs("_build", exist_ok=True)

    for v in [mlflow_version, "latest"]:
        build_docs(v)

    print("Finished building! Output can be found in the `_build` directory")


if __name__ == "__main__":
    main()
