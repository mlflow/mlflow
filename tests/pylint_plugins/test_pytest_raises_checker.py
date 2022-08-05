import pytest

from tests.pylint_plugins.utils import create_message, extract_node, skip_if_pylint_unavailable

pytestmark = skip_if_pylint_unavailable()


@pytest.fixture(scope="module")
def test_case():
    # Ref: https://pylint.pycqa.org/en/latest/how_tos/custom_checkers.html#testing-a-checker
    import pylint.testutils
    from pylint_plugins import PytestRaisesChecker

    class TestPytestRaisesWithoutMatch(pylint.testutils.CheckerTestCase):
        CHECKER_CLASS = PytestRaisesChecker

    test_case = TestPytestRaisesWithoutMatch()
    test_case.setup_method()
    return test_case


def without_raises_bad_cases():
    # Single context manager
    root_node = extract_node(
        """
with pytest.raises(Exception):
    raise Exception("failed")
"""
    )
    yield root_node, root_node.items[0][0], (2, 5)

    # Multiple context managers
    root_node = extract_node(
        """
with context_manager, pytest.raises(Exception):
    raise Exception("failed")
"""
    )
    yield root_node, root_node.items[1][0], (2, 22)

    # Without `with`
    root_node = extract_node(
        """
pytest.raises(Exception)
"""
    )
    yield root_node, root_node, (2, 0)


def without_raises_iter_good_cases():
    # Single context manager
    yield extract_node(
        """
with pytest.raises(Exception, match="failed"):
    raise Exception("failed")
"""
    )

    # Multiple context managers
    yield extract_node(
        """
with context_manager, pytest.raises(Exception, match="failed"):
    raise Exception("failed")
"""
    )

    # Without `with`
    yield extract_node(
        """
pytest.raises(Exception, match="failed")
"""
    )


def test_without_raises(test_case):
    for root_node, error_node, (line, col_offset) in without_raises_bad_cases():
        with test_case.assertAddsMessages(
            create_message(test_case.CHECKER_CLASS.WITHOUT_MATCH, error_node, line, col_offset)
        ):
            test_case.walk(root_node)

    for root_node in without_raises_iter_good_cases():
        with test_case.assertNoMessages():
            test_case.walk(root_node)


def complex_body_bad_cases():
    root_node = extract_node(
        """
with pytest.raises(Exception, match="failed"):
    a = 1
    b = 2
    assert a == b
"""
    )
    yield root_node, root_node, (2, 0)

    root_node = extract_node(
        """
with pytest.raises(Exception, match="failed"), mock_patch("module.function"):
    a = 1
    b = 2
    assert a == b
"""
    )
    yield root_node, root_node, (2, 0)


def complex_body_good_cases():
    yield extract_node(
        """
with pytest.raises(Exception, match="failed"):
    function_that_throws()
"""
    )

    yield extract_node(
        """
with pytest.raises(Exception, match="failed"):
    if condition:
        do_a()
    else:
        do_b()
"""
    )


def test_complex_body(test_case):
    for root_node, error_node, (line, col_offset) in complex_body_bad_cases():
        with test_case.assertAddsMessages(
            create_message(test_case.CHECKER_CLASS.COMPLEX_BODY, error_node, line, col_offset)
        ):
            test_case.walk(root_node)

    for root_node in complex_body_good_cases():
        with test_case.assertNoMessages():
            test_case.walk(root_node)
