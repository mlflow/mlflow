def test_oom():
    lst = []
    while True:
        lst.append(" " * 10**6)
