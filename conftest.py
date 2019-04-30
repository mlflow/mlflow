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
