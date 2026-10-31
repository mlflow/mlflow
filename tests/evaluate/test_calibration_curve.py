import numpy as np
import pytest

from mlflow.models.evaluation.calibration_curve import plot_calibration_curve


@pytest.mark.parametrize(
    "labels",
    [
        [0, 1, 2],
        [1, 2, 3],
        ["setosa", "versicolor", "virginica"],
    ],
)
def test_multiclass_calibration_curve_uses_class_labels(labels):
    # Each sample gets probability 0.9 for its true class and 0.05 for the others, so for
    # every class the 0.05 bin holds no positives and the 0.9 bin holds only positives.
    class_indices = np.repeat(np.arange(3), 20)
    y_probs = np.full((len(class_indices), 3), 0.05)
    y_probs[np.arange(len(class_indices)), class_indices] = 0.9
    y_true = np.array(labels)[class_indices]

    fig = plot_calibration_curve(
        y_true=y_true,
        y_probs=y_probs,
        pos_label=None,
        calibration_config={},
        label_list=np.array(labels),
    )

    class_lines = [line for line in fig.axes[0].lines if line.get_label().startswith("Class ")]
    assert [line.get_label() for line in class_lines] == [f"Class {label}" for label in labels]
    for line in class_lines:
        # x: mean predicted probability, y: fraction of positives (as the axis labels say)
        np.testing.assert_allclose(line.get_xdata(), [0.05, 0.9])
        np.testing.assert_allclose(line.get_ydata(), [0.0, 1.0])
