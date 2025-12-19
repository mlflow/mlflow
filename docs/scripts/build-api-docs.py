import os
import shutil
import subprocess

import click


@click.command()
@click.option("--with-r", "with_r", is_flag=True, default=False, help="Build R documentation")
@click.option(
    "--with-ts", "with_ts", is_flag=True, default=True, help="Build TypeScript documentation"
)
def main(with_r, with_ts):
    try:
        # Run "make rsthtml" in "api_reference" subfolder
        print("Building API reference documentation...")
        subprocess.run(["make", "clean"], check=True, cwd="api_reference")
        subprocess.run(["make", "rsthtml"], check=True, cwd="api_reference")
        subprocess.run(["make", "javadocs"], check=True, cwd="api_reference")
        if with_r:
            subprocess.run(["make", "rdocs"], check=True, cwd="api_reference")
        if with_ts:
            subprocess.run(["make", "tsdocs"], check=True, cwd="api_reference")
        print("Build successful.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        exit(1)

    destination_folder = "static/api_reference"
    source_folder = "api_reference/build/html"

    # Remove the destination folder if it exists
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
        print(f"Removed existing static API docs at {destination_folder}.")

    # Copy the contents of "api_reference/build/html" to "static/api_reference"
    shutil.copytree(source_folder, destination_folder)
    print(f"Copied files from {source_folder} to {destination_folder}.")


if __name__ == "__main__":
    main()
