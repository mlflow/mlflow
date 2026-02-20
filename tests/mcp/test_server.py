from unittest.mock import Mock

import click

from mlflow.mcp.server import fn_wrapper


def test_fn_wrapper_skips_click_unset_for_missing_optional_params():
    capture_order_by = Mock()

    @click.command()
    @click.option("--experiment-id", required=True)
    @click.option("--order-by", type=click.STRING)
    def search_traces(experiment_id: str, order_by: str | None = None) -> None:
        capture_order_by(order_by)

    wrapped = fn_wrapper(search_traces)
    wrapped(experiment_id="0")

    capture_order_by.assert_called_once_with(None)
