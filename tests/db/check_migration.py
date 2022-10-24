import mlflow
import click


@click.group()
def cli():
    pass


@cli.command()
def pre_migration():
    exp_id = mlflow.create_experiment("experiment", tags={"experiment_tag": "value"})
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run():
        pass


@cli.command()
def post_migration():
    experiments = mlflow.search_experiments(filter_string="name = 'experiment'")
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment.name == "experiment"
    assert experiment.tags == {"experiment_tag": "value"}

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.info.experiment_id == experiment.experiment_id


if __name__ == "__main__":
    cli()
