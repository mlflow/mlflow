import os


def pytest_addoption(parser):
    parser.addoption('--large-only', action='store_true', dest="large_only",
                     default=False, help="Run only tests decorated with 'large' annotation")
    parser.addoption('--large', action='store_true', dest="large",
                     default=False, help="Run tests decorated with 'large' annotation")
    parser.addoption('--release', action='store_true', dest="release",
                     default=False, help="Run tests decorated with 'release' annotation")
    parser.addoption("--requires-ssh", action='store_true', dest="requires_ssh",
                     default=False, help="Run tests decorated with 'requires_ssh' annotation. "
                                         "These tests require keys to be configured locally "
                                         "for SSH authentication.")
    parser.addoption("--ignore-flavors", action='store_true', dest="ignore_flavors",
                     default=False, help="Ignore tests for model flavors.")


def pytest_configure(config):
    # Override the markexpr argument to pytest
    # See https://docs.pytest.org/en/latest/example/markers.html for more details
    markexpr = []
    if not config.option.large and not config.option.large_only:
        markexpr.append('not large')
    elif config.option.large_only:
        markexpr.append('large')
    if not config.option.release:
        markexpr.append('not release')
    if not config.option.requires_ssh:
        markexpr.append('not requires_ssh')
    if len(markexpr) > 0:
        setattr(config.option, 'markexpr', " and ".join(markexpr))


def pytest_ignore_collect(path, config):
    if not config.getoption("--ignore-flavors"):
        return False

    flavors_to_ignore = [
        "tests/examples",
        "tests/h2o",
        "tests/keras",
        "tests/pytorch",
        "tests/pyfunc",
        "tests/sagemaker",
        "tests/sklearn",
        "tests/spark",
        "tests/tensorflow",
        "tests/azureml",
        "tests/onnx",
        "tests/keras_autolog",
        "tests/tensorflow_autolog",
        "tests/gluon",
        "tests/gluon_autolog",
        "tests/xgboost",
        "tests/lightgbm",
        "tests/spacy",
        "tests/spark_autologging",
        "tests/models",
    ]

    relpath = os.path.relpath(str(path), config.rootdir)

    return relpath in flavors_to_ignore
