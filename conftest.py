def pytest_addoption(parser):
    parser.addoption('--large', action='store_true', dest="large",
                     default=False, help="Run tests decorated with 'large' annotation")


def pytest_configure(config):
    if not config.option.large:
        setattr(config.option, 'markexpr', 'not large')
