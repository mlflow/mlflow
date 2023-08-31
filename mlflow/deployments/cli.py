import json
import sys
from inspect import signature

import click

from mlflow.deployments import interface
from mlflow.utils import cli_args
from mlflow.utils.proto_json_utils import NumpyEncoder, _get_jsonable_obj


def _user_args_to_dict(user_list):
    # Similar function in mlflow.cli is throwing exception on import
    user_dict = {}
    for s in user_list:
        try:
            # Some configs may contain '=' in the value
            name, value = s.split("=", 1)
        except ValueError as exc:
            # not enough values to unpack
            raise click.BadOptionUsage(
                "config",
                "Config options must be a pair and should be "
                "provided as ``-C key=value`` or "
                "``--config key=value``",
            ) from exc
        if name in user_dict:
            raise click.ClickException(f"Repeated parameter: '{name}'")
        user_dict[name] = value
    return user_dict


installed_targets = list(interface.plugin_store.registry)
if len(installed_targets) > 0:
    supported_targets_msg = "Support is currently installed for deployment to: {targets}".format(
        targets=", ".join(installed_targets)
    )
else:
    supported_targets_msg = (
        "NOTE: you currently do not have support installed for any deployment targets."
    )

target_details = click.option(
    "--target",
    "-t",
    required=True,
    help=f"""
                                   Deployment target URI. Run
                                   `mlflow deployments help --target-name <target-name>` for
                                   more details on the supported URI format and config options
                                   for a given target.
                                   {supported_targets_msg}

                                   See all supported deployment targets and installation
                                   instructions at
                                   https://mlflow.org/docs/latest/plugins.html#community-plugins
                                   """,
)
deployment_name = click.option("--name", "name", required=True, help="Name of the deployment")
optional_deployment_name = click.option("--name", "name", help="Name of the deployment")
parse_custom_arguments = click.option(
    "--config",
    "-C",
    metavar="NAME=VALUE",
    multiple=True,
    help="Extra target-specific config for the model "
    "deployment, of the form -C name=value. See "
    "documentation/help for your deployment target for a "
    "list of supported config options.",
)

parse_input = click.option(
    "--input-path", "-I", required=True, help="Path to input json file for prediction"
)

parse_output = click.option(
    "--output-path",
    "-O",
    help="File to output results to as a JSON file. If not provided, prints output to stdout.",
)

required_endpoint_param = click.option("--endpoint", required=True, help="Name of the endpoint")
optional_endpoint_param = click.option("--endpoint", help="Name of the endpoint")


@click.group(
    "deployments",
    help=f"""
    Deploy MLflow models to custom targets.
    Run `mlflow deployments help --target-name <target-name>` for
    more details on the supported URI format and config options for a given target.
    {supported_targets_msg}

    See all supported deployment targets and installation instructions in
    https://mlflow.org/docs/latest/plugins.html#community-plugins

    You can also write your own plugin for deployment to a custom target. For instructions on
    writing and distributing a plugin, see
    https://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins.
""",
)
def commands():
    """
    Deploy MLflow models to custom targets. Support is currently installed for
    the following targets: {targets}. Run `mlflow deployments help --target-name <target-name>` for
    more details on the supported URI format and config options for a given target.

    To deploy to other targets, you must first install an
    appropriate third-party Python plugin. See the list of known community-maintained plugins
    at https://mlflow.org/docs/latest/plugins.html#community-plugins.

    You can also write your own plugin for deployment to a custom target. For instructions on
    writing and distributing a plugin, see
    https://mlflow.org/docs/latest/plugins.html#writing-your-own-mlflow-plugins.
    """


@commands.command("create")
@optional_endpoint_param
@parse_custom_arguments
@deployment_name
@target_details
@cli_args.MODEL_URI
@click.option(
    "--flavor",
    "-f",
    help="Which flavor to be deployed. This will be auto inferred if it's not given",
)
def create_deployment(flavor, model_uri, target, name, config, endpoint):
    """
    Deploy the model at ``model_uri`` to the specified target.

    Additional plugin-specific arguments may also be passed to this command, via `-C key=value`
    """
    config_dict = _user_args_to_dict(config)
    client = interface.get_deploy_client(target)

    sig = signature(client.create_deployment)
    if "endpoint" in sig.parameters:
        deployment = client.create_deployment(
            name, model_uri, flavor, config=config_dict, endpoint=endpoint
        )
    else:
        deployment = client.create_deployment(name, model_uri, flavor, config=config_dict)
    click.echo("\n{} deployment {} is created".format(deployment["flavor"], deployment["name"]))


@commands.command("update")
@optional_endpoint_param
@parse_custom_arguments
@deployment_name
@target_details
@click.option(
    "--model-uri",
    "-m",
    default=None,
    metavar="URI",
    help="URI to the model. A local path, a 'runs:/' URI, or a"
    " remote storage URI (e.g., an 's3://' URI). For more information"
    " about supported remote URIs for model artifacts, see"
    " https://mlflow.org/docs/latest/tracking.html"
    "#artifact-stores",
)
@click.option(
    "--flavor",
    "-f",
    help="Which flavor to be deployed. This will be auto inferred if it's not given",
)
def update_deployment(flavor, model_uri, target, name, config, endpoint):
    """
    Update the deployment with ID `deployment_id` in the specified target.
    You can update the URI of the model and/or the flavor of the deployed model (in which case the
    model URI must also be specified).

    Additional plugin-specific arguments may also be passed to this command, via `-C key=value`.
    """
    config_dict = _user_args_to_dict(config)
    client = interface.get_deploy_client(target)

    sig = signature(client.update_deployment)
    if "endpoint" in sig.parameters:
        ret = client.update_deployment(
            name, model_uri=model_uri, flavor=flavor, config=config_dict, endpoint=endpoint
        )
    else:
        ret = client.update_deployment(name, model_uri=model_uri, flavor=flavor, config=config_dict)
    click.echo("Deployment {} is updated (with flavor {})".format(name, ret["flavor"]))


@commands.command("delete")
@optional_endpoint_param
@parse_custom_arguments
@deployment_name
@target_details
def delete_deployment(target, name, config, endpoint):
    """
    Delete the deployment with name given at `--name` from the specified target.
    """
    client = interface.get_deploy_client(target)

    sig = signature(client.delete_deployment)
    if "config" in sig.parameters:
        config_dict = _user_args_to_dict(config)
        if "endpoint" in sig.parameters:
            client.delete_deployment(name, config=config_dict, endpoint=endpoint)
        else:
            client.delete_deployment(name, config=config_dict)
    else:
        if "endpoint" in sig.parameters:
            client.delete_deployment(name, endpoint=endpoint)
        else:
            client.delete_deployment(name)

    click.echo(f"Deployment {name} is deleted")


@commands.command("list")
@optional_endpoint_param
@target_details
def list_deployment(target, endpoint):
    """
    List the names of all model deployments in the specified target. These names can be used with
    the `delete`, `update`, and `get` commands.
    """
    client = interface.get_deploy_client(target)

    sig = signature(client.list_deployments)
    if "endpoint" in sig.parameters:
        ids = client.list_deployments(endpoint=endpoint)
    else:
        ids = client.list_deployments()
    click.echo(f"List of all deployments:\n{ids}")


@commands.command("get")
@optional_endpoint_param
@deployment_name
@target_details
def get_deployment(target, name, endpoint):
    """
    Print a detailed description of the deployment with name given at ``--name`` in the specified
    target.
    """
    client = interface.get_deploy_client(target)

    sig = signature(client.get_deployment)
    if "endpoint" in sig.parameters:
        desc = client.get_deployment(name, endpoint=endpoint)
    else:
        desc = client.get_deployment(name)
    for key, val in desc.items():
        click.echo(f"{key}: {val}")
    click.echo("\n")


@commands.command("help")
@target_details
def target_help(target):
    """
    Display additional help for a specific deployment target, e.g. info on target-specific config
    options and the target's URI format.
    """
    click.echo(interface._target_help(target))


@commands.command("run-local")
@parse_custom_arguments
@deployment_name
@target_details
@cli_args.MODEL_URI
@click.option(
    "--flavor",
    "-f",
    help="Which flavor to be deployed. This will be auto inferred if it's not given",
)
def run_local(flavor, model_uri, target, name, config):
    """
    Deploy the model locally. This has very similar signature to ``create`` API
    """
    config_dict = _user_args_to_dict(config)
    interface.run_local(target, name, model_uri, flavor, config_dict)


def predictions_to_json(raw_predictions, output):
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")
    json.dump(predictions, output, cls=NumpyEncoder)


@commands.command("predict")
@click.option(
    "--name",
    "name",
    help="Name of the deployment. Exactly one of --name or --endpoint must be specified.",
)
@click.option(
    "--endpoint",
    help="Name of the endpoint. Exactly one of --name or --endpoint must be specified.",
)
@target_details
@parse_input
@parse_output
def predict(target, name, input_path, output_path, endpoint):
    """
    Predict the results for the deployed model for the given input(s)
    """
    import pandas as pd

    if (name, endpoint).count(None) != 1:
        raise click.UsageError("Must specify exactly one of --name or --endpoint.")

    df = pd.read_json(input_path)
    client = interface.get_deploy_client(target)

    sig = signature(client.predict)
    if "endpoint" in sig.parameters:
        result = client.predict(name, df, endpoint=endpoint)
    else:
        result = client.predict(name, df)
    if output_path is not None:
        result.to_json(output_path)
    else:
        click.echo(result.to_json())


@commands.command("explain")
@click.option(
    "--name",
    "name",
    help="Name of the deployment. Exactly one of --name or --endpoint must be specified.",
)
@click.option(
    "--endpoint",
    help="Name of the endpoint. Exactly one of --name or --endpoint must be specified.",
)
@target_details
@parse_input
@parse_output
def explain(target, name, input_path, output_path, endpoint):
    """
    Generate explanations of model predictions on the specified input for
    the deployed model for the given input(s). Explanation output formats vary
    by deployment target, and can include details like feature importance for
    understanding/debugging predictions. Run `mlflow deployments help` or
    consult the documentation for your plugin for details on explanation format.
    For information about the input data formats accepted by this function,
    see the following documentation:
    https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools
    """
    import pandas as pd

    if (name, endpoint).count(None) != 1:
        raise click.UsageError("Must specify exactly one of --name or --endpoint.")

    df = pd.read_json(input_path)
    client = interface.get_deploy_client(target)

    sig = signature(client.explain)
    if "endpoint" in sig.parameters:
        result = client.explain(name, df, endpoint=endpoint)
    else:
        result = client.explain(name, df)
    if output_path:
        with open(output_path, "w") as fp:
            predictions_to_json(result, fp)
    else:
        predictions_to_json(result, sys.stdout)


@commands.command("create-endpoint")
@click.option(
    "--config",
    "-C",
    metavar="NAME=VALUE",
    multiple=True,
    help="Extra target-specific config for the endpoint, "
    "of the form -C name=value. See "
    "documentation/help for your deployment target for a "
    "list of supported config options.",
)
@required_endpoint_param
@target_details
def create_endpoint(target, name, config):
    """
    Create an endpoint with the specified name at the specified target.

    Additional plugin-specific arguments may also be passed to this command, via `-C key=value`
    """
    config_dict = _user_args_to_dict(config)
    client = interface.get_deploy_client(target)
    endpoint = client.create_endpoint(name, config=config_dict)
    click.echo("\nEndpoint {} is created".format(endpoint["name"]))


@commands.command("update-endpoint")
@click.option(
    "--config",
    "-C",
    metavar="NAME=VALUE",
    multiple=True,
    help="Extra target-specific config for the endpoint, "
    "of the form -C name=value. See "
    "documentation/help for your deployment target for a "
    "list of supported config options.",
)
@required_endpoint_param
@target_details
def update_endpoint(target, endpoint, config):
    """
    Update the specified endpoint at the specified target.

    Additional plugin-specific arguments may also be passed to this command, via `-C key=value`
    """
    config_dict = _user_args_to_dict(config)
    client = interface.get_deploy_client(target)
    client.update_endpoint(endpoint, config=config_dict)
    click.echo(f"\nEndpoint {endpoint} is updated")


@commands.command("delete-endpoint")
@required_endpoint_param
@target_details
def delete_endpoint(target, endpoint):
    """
    Delete the specified endpoint at the specified target
    """
    client = interface.get_deploy_client(target)
    client.delete_endpoint(endpoint)
    click.echo(f"\nEndpoint {endpoint} is deleted")


@commands.command("list-endpoints")
@target_details
def list_endpoints(target):
    """
    List all endpoints at the specified target
    """
    client = interface.get_deploy_client(target)
    ids = client.list_endpoints()
    click.echo(f"List of all endpoints:\n{ids}")


@commands.command("get-endpoint")
@required_endpoint_param
@target_details
def get_endpoint(target, endpoint):
    """
    Get details for the specified endpoint at the specified target
    """
    client = interface.get_deploy_client(target)
    desc = client.get_endpoint(endpoint)
    for key, val in desc.items():
        click.echo(f"{key}: {val}")
    click.echo("\n")
