import contextlib
import logging
import os
import threading
import time
import webbrowser
from collections.abc import Generator
from pathlib import Path

import click

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
    help="Tracking URI of an existing MLflow server to populate with demo data.",
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

    use_existing = click.confirm(
        click.style("Do you have an MLflow server already running?", fg="bright_blue"),
        default=False,
    )

    if use_existing:
        return click.prompt(
            click.style("Enter the tracking server URL", fg="bright_blue"),
            default="http://localhost:5000",
        )
    return None


def _run_with_existing_server(
    tracking_uri: str, no_browser: bool, debug: bool, refresh: bool
) -> None:
    import mlflow
    from mlflow.demo import generate_all_demos
    from mlflow.demo.base import DEMO_EXPERIMENT_NAME

    click.echo()
    click.echo(f"Connecting to MLflow server at {tracking_uri}...")

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

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    if experiment is None:
        raise click.ClickException(
            f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found. "
            "This should not happen after generating demo data."
        )
    experiment_url = f"{tracking_uri.rstrip('/')}/#/experiments/{experiment.experiment_id}/overview"

    click.echo()
    click.secho(f"View the demo at: {experiment_url}", fg="green", bold=True)

    if not no_browser:
        click.echo()
        click.echo("Opening the MLflow UI...")
        webbrowser.open(experiment_url)


def _run_with_new_server(port: int | None, no_browser: bool, debug: bool, refresh: bool) -> None:
    import mlflow
    from mlflow.demo import generate_all_demos
    from mlflow.demo.base import DEMO_EXPERIMENT_NAME
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

    experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    if experiment is None:
        raise click.ClickException(
            f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found. "
            "This should not happen after generating demo data."
        )
    experiment_url = f"http://127.0.0.1:{port}/#/experiments/{experiment.experiment_id}/overview"

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
