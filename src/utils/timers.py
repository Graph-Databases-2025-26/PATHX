import time
from contextlib import contextmanager

@contextmanager
def timer():
    t0 = time.time()
    try:
        yield lambda: time.time() - t0
    finally:
        ...
