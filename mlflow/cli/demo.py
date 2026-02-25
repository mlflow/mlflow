import contextlib
import logging
import os
import threading
import time
import webbrowser
from collections.abc import Generator
from pathlib import Path
from urllib.parse import urljoin

import click

from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import is_databricks_uri

NOISY_LOGGERS = [
    "alembic",
    "mlflow.store",
    "mlflow.tracking",
    "mlflow.tracing",
    "mlflow.genai",
    "mlflow.server",
    "httpx",
    "httpcore",
    "urllib3",
    "uvicorn",
    "huey",
]


@contextlib.contextmanager
def _suppress_noisy_logs() -> Generator[None, None, None]:
    original_levels: dict[str, int] = {}
    try:
        for logger_name in NOISY_LOGGERS:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)
        yield
    finally:
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


def _set_quiet_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set environment variable so MLflow configures logging in subprocesses
    # This affects mlflow, alembic, and huey loggers via _configure_mlflow_loggers
    os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"


def _check_server_connection(tracking_uri: str, max_retries: int = 3, timeout: int = 5) -> None:
    """Check if the MLflow tracking server is reachable.

    Args:
        tracking_uri: URL of the tracking server.
        max_retries: Maximum number of connection attempts.
        timeout: Timeout in seconds for each connection attempt.

    Raises:
        click.ClickException: If the server is not reachable after all retries.
    """
    import requests

    from mlflow.utils.request_utils import _get_http_response_with_retries

    health_url = urljoin(tracking_uri.rstrip("/") + "/", "health")

    try:
        response = _get_http_response_with_retries(
            method="GET",
            url=health_url,
            max_retries=max_retries,
            backoff_factor=1,
            backoff_jitter=0.5,
            retry_codes=(408, 429, 500, 502, 503, 504),
            timeout=timeout,
            raise_on_status=False,
        )
        response.close()
    except requests.exceptions.ConnectionError as e:
        raise click.ClickException(
            f"Cannot connect to MLflow server at {tracking_uri}\n"
            f"Error: {e}\n\n"
            f"Please verify:\n"
            f"  1. The server is running\n"
            f"  2. The URL is correct\n"
            f"  3. No firewall is blocking the connection"
        ) from None
    except requests.exceptions.Timeout:
        raise click.ClickException(
            f"Connection to MLflow server at {tracking_uri} timed out.\n\n"
            f"Please verify the server is running and responsive."
        ) from None
    except requests.exceptions.RequestException as e:
        raise click.ClickException(
            f"Failed to connect to MLflow server at {tracking_uri}\nError: {e}"
        ) from None


def _check_databricks_connection(tracking_uri: str) -> str:
    """Validate Databricks credentials and return the workspace host URL.

    Args:
        tracking_uri: A Databricks tracking URI (e.g., "databricks" or "databricks://profile").

    Returns:
        The resolved workspace host URL (e.g., "https://my-workspace.databricks.com").

    Raises:
        click.ClickException: If credentials are missing or invalid.
    """
    try:
        host_creds = get_databricks_host_creds(tracking_uri)
        return host_creds.host.rstrip("/")
    except Exception as e:
        raise click.ClickException(
            f"Cannot connect to Databricks workspace.\n"
            f"Error: {e}\n\n"
            f"Please verify:\n"
            f"  1. DATABRICKS_HOST and DATABRICKS_TOKEN environment variables are set, or\n"
            f"  2. A Databricks CLI profile is configured\n\n"
            f"Example:\n"
            f"  export DATABRICKS_HOST='https://your-workspace.databricks.com'\n"
            f"  export DATABRICKS_TOKEN='your-token'\n"
            f"  mlflow demo --tracking-uri databricks"
        ) from None


def _prompt_for_uc_schema() -> str:
    """Ask the user for a Unity Catalog catalog.schema.

    Required on Databricks — prompts and datasets are stored as UC objects.
    """
    click.echo()
    click.echo("Demo data on Databricks requires a Unity Catalog schema (catalog.schema).")
    uc_schema = click.prompt(
        click.style(
            "Enter catalog.schema (e.g., main.default)",
            fg="bright_blue",
        ),
    ).strip()

    if "." not in uc_schema or len(uc_schema.split(".")) != 2:
        raise click.ClickException(
            f"Invalid Unity Catalog schema format: '{uc_schema}'. "
            "Expected 'catalog.schema' (e.g., 'main.default')."
        )
    return uc_schema


@click.command()
@click.option(
    "--port",
    default=None,
    type=int,
    help="Port to run demo server on (only used when starting a new server).",
)
@click.option(
    "--tracking-uri",
    default=None,
    help="Tracking URI of an existing MLflow server or Databricks workspace (e.g., 'databricks').",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't automatically open browser to demo experiment.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable verbose logging output.",
)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    help="Force regenerate demo data by deleting existing data first.",
)
def demo(
    port: int | None,
    tracking_uri: str | None,
    no_browser: bool,
    debug: bool,
    refresh: bool,
) -> None:
    """Launch MLflow with pre-populated demo data for exploring GenAI features.

    By default, creates a persistent environment in ./mlflow-demo/ with SQLite database
    and file-based artifacts, generates demo data, and opens the browser to the demo
    experiment. Data persists across restarts; use --refresh to regenerate.

    To populate an existing MLflow server with demo data, use --tracking-uri:

    mlflow demo                                       # Launch new demo server
    mlflow demo --no-browser                          # Launch without opening browser
    mlflow demo --port 5001                           # Use custom port
    mlflow demo --tracking-uri http://localhost:5000  # Use existing server
    mlflow demo --tracking-uri databricks             # Use Databricks workspace
    """
    if tracking_uri is None:
        tracking_uri = _get_tracking_uri_interactive(port)

    if tracking_uri is None:
        _run_with_new_server(port, no_browser, debug, refresh)
    else:
        _run_with_existing_server(tracking_uri, no_browser, debug, refresh)


def _get_tracking_uri_interactive(port: int | None) -> str | None:
    click.echo()
    click.secho("MLflow Demo Setup", fg="cyan", bold=True)
    click.echo()
    click.echo("Where would you like to run the demo?")
    click.echo()
    click.echo("  1. Start a new local server (default)")
    click.echo("  2. Connect to an existing MLflow server")
    click.echo("  3. Connect to a Databricks workspace")
    click.echo()

    choice = click.prompt(
        click.style("Enter your choice", fg="bright_blue"),
        type=click.IntRange(1, 3),
        default=1,
    )

    if choice == 2:
        return click.prompt(
            click.style("Enter the tracking server URL", fg="bright_blue"),
            default="http://localhost:5000",
        )
    elif choice == 3:
        return "databricks"
    return None


def _run_with_existing_server(
    tracking_uri: str, no_browser: bool, debug: bool, refresh: bool
) -> None:
    import mlflow
    from mlflow.demo import generate_all_demos, set_uc_schema
    from mlflow.demo.base import get_demo_experiment_name

    click.echo()

    if is_databricks_uri(tracking_uri):
        click.echo(f"Connecting to Databricks workspace ({tracking_uri})... ", nl=False)
        host_url = _check_databricks_connection(tracking_uri)
        click.secho("connected!", fg="green")

        uc_schema = _prompt_for_uc_schema()
        set_uc_schema(uc_schema)
    else:
        click.echo(f"Connecting to MLflow server at {tracking_uri}... ", nl=False)
        _check_server_connection(tracking_uri)
        click.secho("connected!", fg="green")

    mlflow.set_tracking_uri(tracking_uri)

    click.echo("Generating demo data... ", nl=False)
    if debug:
        results = generate_all_demos(refresh=refresh)
    else:
        with _suppress_noisy_logs():
            results = generate_all_demos(refresh=refresh)
    click.secho("done!", fg="green")

    if results:
        click.echo(f"  Generated: {', '.join(r.feature for r in results)}")
    else:
        click.echo("  Demo data already exists (skipped generation).")

    demo_name = get_demo_experiment_name()
    experiment = mlflow.get_experiment_by_name(demo_name)
    if experiment is None:
        raise click.ClickException(
            f"Demo experiment '{demo_name}' not found. "
            "This should not happen after generating demo data."
        )

    if is_databricks_uri(tracking_uri):
        experiment_url = f"{host_url}/ml/experiments/{experiment.experiment_id}/traces"
    else:
        experiment_url = (
            f"{tracking_uri.rstrip('/')}/#/experiments/{experiment.experiment_id}/traces"
        )

    click.echo()
    click.secho(f"View the demo at: {experiment_url}", fg="green", bold=True)

    if not no_browser:
        click.echo()
        click.echo("Opening the MLflow UI...")
        webbrowser.open(experiment_url)


def _run_with_new_server(port: int | None, no_browser: bool, debug: bool, refresh: bool) -> None:
    import mlflow
    from mlflow.demo import generate_all_demos
    from mlflow.demo.base import get_demo_experiment_name
    from mlflow.server import _run_server
    from mlflow.server.handlers import initialize_backend_stores
    from mlflow.utils import find_free_port, is_port_available

    # Suppress noisy logs early (before any initialization) unless debug mode
    if not debug:
        _set_quiet_logging()

    if port is None:
        port = find_free_port()
    elif not is_port_available(port):
        raise click.ClickException(
            f"Port {port} is already in use. "
            f"Either stop the process using that port, "
            f"or run: mlflow demo --port <DIFFERENT_PORT>"
        )

    demo_dir = Path.cwd() / "mlflow-demo"
    demo_dir.mkdir(exist_ok=True)

    db_path = demo_dir / "mlflow.db"
    artifact_path = demo_dir / "artifacts"
    artifact_path.mkdir(exist_ok=True)

    backend_uri = f"sqlite:///{db_path}"
    artifact_uri = artifact_path.as_uri()

    os.environ["MLFLOW_TRACKING_URI"] = backend_uri

    click.echo()
    click.echo("Initializing demo environment... ", nl=False)
    initialize_backend_stores(backend_uri, backend_uri, artifact_uri)
    click.secho("done!", fg="green")

    click.echo("Generating demo data... ", nl=False)
    results = generate_all_demos(refresh=refresh)
    click.secho("done!", fg="green")

    if results:
        click.echo(f"  Generated: {', '.join(r.feature for r in results)}")

    demo_name = get_demo_experiment_name()
    experiment = mlflow.get_experiment_by_name(demo_name)
    if experiment is None:
        raise click.ClickException(
            f"Demo experiment '{demo_name}' not found. "
            "This should not happen after generating demo data."
        )
    experiment_url = f"http://127.0.0.1:{port}/#/experiments/{experiment.experiment_id}/traces"

    if not no_browser:

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(experiment_url)

        threading.Thread(target=open_browser, daemon=True, name="DemoBrowserOpener").start()

    click.echo()
    click.secho(f"MLflow Tracking Server running at: http://127.0.0.1:{port}", fg="green")
    click.secho(f"View the demo at: {experiment_url}", fg="green", bold=True)
    click.echo()
    click.echo("Press Ctrl+C to stop the server.")
    click.echo()

    _run_server(
        file_store_path=backend_uri,
        registry_store_uri=backend_uri,
        default_artifact_root=artifact_uri,
        serve_artifacts=True,
        artifacts_only=False,
        artifacts_destination=None,
        host="127.0.0.1",
        port=port,
        workers=1,
        uvicorn_opts="--log-level warning" if not debug else None,
    )
