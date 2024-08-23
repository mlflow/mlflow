import subprocess

with subprocess.Popen(["bin/start-uc-server"], cwd="unitycatalog") as proc:
    # run tests here
    proc.terminate()
