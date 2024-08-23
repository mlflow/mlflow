import subprocess
import time

with subprocess.Popen(["bin/start-uc-server"], cwd="unitycatalog") as proc:
    time.sleep(10)
    # run tests here
    proc.terminate()
