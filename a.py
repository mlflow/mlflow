import mlflow


for name in "ab":
    if not mlflow.get_experiment_by_name(name):
        mlflow.create_experiment(name)


def log():
    for i in range(10):
        print(i)
        with mlflow.start_run():
            mlflow.log_param(str(i), i)


log()


mlflow.set_experiment("a")

log()

mlflow.set_experiment("b")
log()
