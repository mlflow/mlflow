import subprocess

with subprocess.Popen(["bin/start-uc-server"]) as proc:
    proc.terminate()
