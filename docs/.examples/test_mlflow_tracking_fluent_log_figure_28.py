# Location: mlflow/tracking/fluent.py:998
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:998 '])
def test(_):
    import mlflow
    from plotly import graph_objects as go

    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

    with mlflow.start_run():
        mlflow.log_figure(fig, "figure.html")


if __name__ == "__main__":
    test()
