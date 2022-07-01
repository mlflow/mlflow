import argparse
import sys
import yaml


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pip-compatible requirements.txt file from an MLflow"
            " requirements.yaml specification"
        )
    )
    parser.add_argument(
        "--requirements-yaml-location",
        required=True,
        help="Local file path of the requirements.yaml specification.",
    )
    parser.add_argument(
        "--requirements-txt-output-location",
        required=True,
        help="Local output path for the generated requirements.txt file.",
    )

    return parser.parse_args(args)


def validate_requirements_yaml(requirements_yaml):
    assert isinstance(
        requirements_yaml, dict
    ), "requirements.yaml contents must be a YAML dictionary"
    for package_entry in requirements_yaml.values():
        assert isinstance(
            package_entry, dict
        ), f"Entry in requirements.yaml does not have required dictionary structure: {package_entry}"
        pip_release = package_entry.get("pip_release")
        assert pip_release is not None and isinstance(
            pip_release, str
        ), f"Entry in requirements.yaml does not define a valid 'pip_release' string value: {package_entry}"
        max_major_version = package_entry.get("max_major_version")
        assert max_major_version is not None and isinstance(
            max_major_version, int
        ), f"Entry in requirements.yaml does not specify a valid 'max_major_version' integer value: {package_entry}"
        if "minimum" in package_entry:
            assert isinstance(
                package_entry["minimum"], str
            ), f"Entry in requirements.yaml contains an invalid 'minimum' version string specification: {package_entry}"
        if "unsupported" in package_entry:
            assert isinstance(package_entry["unsupported"], list) and all(
                [
                    isinstance(unsupported_entry, str)
                    for unsupported_entry in package_entry["unsupported"]
                ]
            ), "Entry in requirements.yaml contains an invalid 'unsupported' versions specification. Unsupported versions should be specified as lists of strings: {package_entry}"
        if "markers" in package_entry:
            assert isinstance(
                package_entry["markers"], str
            ), f"Entry in requirements.yaml contains invalid 'markers' string specification: {package_entry}"


def generate_requirements_txt_content(requirements_yaml):
    requirement_strs = []
    for package_entry in requirements_yaml.values():
        pip_release = package_entry["pip_release"]
        version_specs = []

        max_major_version = package_entry["max_major_version"]
        version_specs += [f"<{max_major_version + 1}.*"]

        min_version = package_entry.get("minimum")
        version_specs += [f">={min_version}"] if min_version else []

        unsupported_versions = package_entry.get("unsupported", [])
        version_specs += [f"!={version}" for version in unsupported_versions]

        markers = package_entry.get("markers")
        markers = f"; {markers}" if markers else ""

        requirement_str = f"{pip_release}{','.join(version_specs)}{markers}"
        requirement_strs.append(requirement_str)

    return "\n".join(requirement_strs)


def main(args):
    args = parse_args(args)
    with open(args.requirements_yaml_location, "r") as f:
        requirements_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    validate_requirements_yaml(requirements_yaml)
    requirements_txt_content = generate_requirements_txt_content(requirements_yaml)
    with open(args.requirements_txt_output_location, "w") as f:
        f.write(requirements_txt_content)


if __name__ == "__main__":
    main(sys.argv[1:])
