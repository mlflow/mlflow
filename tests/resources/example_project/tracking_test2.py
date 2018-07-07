import mlflow
TEST_PROJECT_DIR = "/Users/sid/code/mlflow/tests/resources/example_project"


def launch_run():
    return mlflow.projects.run(
        TEST_PROJECT_DIR, entry_point="sleep",
        parameters={"duration": 120},
        use_conda=False, experiment_id=0, block=False)


if __name__ == "__main__":
    import os
    print("Current pid %s" % os.getpid())
    runs = []
    for i in range(3):
        runs.append(launch_run())
    import time
    time.sleep(1)
    raise Exception("Exception in parent")
    # CTRL+Cing should kill the monitoring subprocesses & the command processes, but it just
    # kills the monitoring ones. A thought: what if instead of subprocess.Popen we

