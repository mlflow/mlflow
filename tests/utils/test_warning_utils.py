import warnings

from mlflow.utils.warnings_utils import suppress_warnings_containing


def test_suppress_warnings_containing():
    @suppress_warnings_containing("test_pattern")
    def test_function():
        warnings.warn("This is a test warning with 'test_pattern''", UserWarning)
        warnings.warn("This is another warning without the string", UserWarning)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        test_function()
        assert len(recorded_warnings) == 1
        assert str(recorded_warnings[0].message) == "This is another warning without the string"
