from unittest import mock


def test_a():
    m = mock.MagicMock()
    m(1)
    m(2)
    m(3)
    m(4)
    m.assert_has_calls(
        [
            mock.call(1),
            mock.call(2),
            mock.call(3),
        ]
    )  # succeeds
    assert m.mock_calls == [
        mock.call(1),
        mock.call(2),
        mock.call(3),
    ]  # fails
