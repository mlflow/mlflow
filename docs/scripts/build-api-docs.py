import os
import shutil
import subprocess


def build_and_copy_docs():
    try:
        # Run "make rsthtml" in "api_reference" subfolder
        print("Building API reference documentation...")
        subprocess.run(["make", "clean"], check=True, cwd="api_reference")
        subprocess.run(["make", "rsthtml"], check=True, cwd="api_reference")
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


build_and_copy_docs()
