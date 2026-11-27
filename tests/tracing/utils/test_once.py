from mlflow.tracing.utils.once import Once


def test_once_done_reflects_whether_the_function_ran():
    once = Once()
    calls = []

    assert not once.done
    once.do_once(lambda: calls.append(1))
    assert once.done
    once.do_once(lambda: calls.append(2))
    assert once.done
    assert calls == [1]
