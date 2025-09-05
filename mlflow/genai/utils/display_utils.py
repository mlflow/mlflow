import sys

from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracing.display.display_handler import _is_jupyter
from mlflow.tracking._tracking_service.utils import _get_store

_EVAL_OUTPUT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Evaluation output</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}

        .header {{
            a.button {{
                padding: 4px 8px;
                line-height: 20px;
                box-shadow: none;
                height: 20px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                vertical-align: middle;
                background-color: rgb(34, 114, 180);
                color: rgb(255, 255, 255);
                text-decoration: none;
                animation-duration: 0s;
                transition: none 0s ease 0s;
                position: relative;
                white-space: nowrap;
                text-align: center;
                border: 1px solid rgb(192, 205, 216);
                cursor: pointer;
                user-select: none;
                touch-action: manipulation;
                border-radius: 4px;
                gap: 6px;
            }}

            a.button:hover {{
                background-color: rgb(14, 83, 139) !important;
                border-color: transparent !important;
                color: rgb(255, 255, 255) !important;
            }}
        }}

        .warnings-section {{
            margin-top: 8px;

            ul {{
                list-style-type: none;
            }}
        }}

        .instructions-section {{
            margin-top: 16px;
            font-size: 14px;

            ul {{
                margin-top: 0;
                margin-bottom: 0;
            }}
        }}

        code {{
            font-family: monospace;
        }}

        .note {{
            color: #666;
        }}

        a {{
            color: #2272B4;
            text-decoration: none;
        }}

        a:hover {{
            color: #005580;
        }}
    </style>
</head>
<body>
<div>
    <div class="header">
        <a href="{eval_results_url}" class="button">
            View evaluation results in MLflow
            <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="none" viewBox="0 0 16 16" aria-hidden="true" focusable="false" class="">
                <path fill="currentColor" d="M10 1h5v5h-1.5V3.56L8.53 8.53 7.47 7.47l4.97-4.97H10z"></path>
                <path fill="currentColor" d="M1 2.75A.75.75 0 0 1 1.75 2H8v1.5H2.5v10h10V8H14v6.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75z"></path>
            </svg>
        </a>
    </div>
</div>
</body>
</html>
"""  # noqa: E501

_NON_IPYTHON_OUTPUT_TEXT = """
✨ Evaluation completed.

Metrics and evaluation results are logged to the MLflow run:
  Run name: \033[94m{run_name}\033[0m
  Run ID: \033[94m{run_id}\033[0m
"""


def display_evaluation_output(run_id: str):
    """
    Displays summary of the evaluation result, errors and warnings if any,
    and instructions on what to do after running `mlflow.evaluate`.

    TODO: This function only works for OSS tracking server, does not resolve
    Databricks workspace URL. When we migrate Databricks users to OSS harness,
    update this logic to resolve the workspace URL.
    """
    store = _get_store()
    run = store.get_run(run_id)

    if not isinstance(store, RestStore):
        # Cannot determine the host URL if the server is not remote.
        # Print a general guidance instead.
        sys.stdout.write(_NON_IPYTHON_OUTPUT_TEXT.format(run_name=run.info.run_name, run_id=run_id))
        sys.stdout.write("""
To view the detailed evaluation results with sample-wise scores,
open the \033[93m\033[1mTraces\033[0m tab in the Run page in the MLflow UI.\n\n""")
        return

    host_url = store.get_host_creds().host.rstrip("/")
    experiment_id = run.info.experiment_id
    # Navigate to 'traces' tab that shows assessments and aggregations.
    uri = f"{host_url}/#/experiments/{experiment_id}/runs/{run_id}/traces"

    if _is_jupyter():
        from IPython.display import HTML, display

        display(HTML(_EVAL_OUTPUT_HTML.format(eval_results_url=uri)))
    else:
        sys.stdout.write(_NON_IPYTHON_OUTPUT_TEXT.format(run_name=run.info.run_name, run_id=run_id))
        sys.stdout.write(f"View the evaluation results at \033[93m{uri}\033[0m\n\n")
