# Import a dependency in MLflow's setup.py that's not included by default in conda environments,
# verify that it fails

import argparse
import os


def main(expected_env_name):
    try:
        actual_conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
        assert (
            actual_conda_env == expected_env_name
        ), "Script expected to be run from conda env %s but was actually run from env %s" % (
            expected_env_name,
            actual_conda_env,
        )
        import gunicorn  # pylint: disable=unused-import
    except ImportError:
        return
    raise Exception("Expected exception when attempting to import gunicorn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conda-env-name",
        help="Name of the conda environment this script should expect to be run from",
        type=str,
    )
    args = parser.parse_args()
    main(args.conda_env_name)
