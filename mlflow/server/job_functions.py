
import time

def job_func1(x, y):
    time.sleep(1)
    return {
        'a': x + y,
        'b': x * y,
    }

