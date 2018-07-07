import sys
import threading
import time
import os

def fn():
    try:
        time.sleep(3)
    except SystemExit as e:
        print("Sup fam caught a systemexit %s" % e)
        raise
    print("finished fn()")

t = threading.Thread(target=fn)
t.start()
print("In main thread, exiting now...")
sys.exit(1)
