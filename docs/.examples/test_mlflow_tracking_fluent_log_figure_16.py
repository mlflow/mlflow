# Location: mlflow/tracking/fluent.py:986
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/fluent.py:986 '])
def test(_):
    import mlflow
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])

    with mlflow.start_run():
        mlflow.log_figure(fig, "figure.png")


if __name__ == "__main__":
    test()
