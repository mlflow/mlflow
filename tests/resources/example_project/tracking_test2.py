# TODO: delete this script/turn it into unit tests
import mlflow
GIT_PROJECT_URI = "https://github.com/databricks/mlflow-example"
TEST_PROJECT_DIR = "/Users/sid/code/mlflow/tests/resources/example_project"


def launch_run():
    return mlflow.projects.run(
        TEST_PROJECT_DIR, entry_point="sleep",
        parameters={"duration": 30},
        use_conda=False, experiment_id=0, block=False)


def launch_fail_run():
    return mlflow.projects.run(
        TEST_PROJECT_DIR, entry_point="sleep", parameters={"alpha": "0.4", "duration": "3"},
        use_conda=False, block=False)


def launch_databricks_run():
    return mlflow.projects.run(
        GIT_PROJECT_URI, parameters={"alpha": "0.4"}, use_conda=False, block=False,
        cluster_spec="/Users/sid/code/misc/cluster_spec.json", mode="databricks")


if __name__ == "__main__":
    import os
    print("Current pid %s" % os.getpid())
    runs = []
    for i in range(3):
        runs.append(launch_run())
        runs.append(launch_fail_run())
    launch_databricks_run()
    # time.sleep(1)
    # print("Waiting on run %s" % runs[0]._active_run.run_info.entry_point_name)
    runs[0].wait()
    # raise Exception("yo")

