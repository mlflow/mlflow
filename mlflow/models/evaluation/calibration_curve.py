from matplotlib.axes import Axes
from sklearn.calibration import CalibrationDisplay


def plot_calibration_curve(
    y_true,
    y_probs,
    pos_label,
    calibration_config,
) -> Axes:
    """Generate a calibration curve for a trained classifier

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probs (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        pos_label (str): Label for the positive class.

        calibration_config (dict[str, Union[str, int]]):
            Additional parameters for sklearn.CalibrationDisplay

    Raises:
        TypeError: if calibration_config["calibration_classifier_name"] is not str
        TypeError: if calibration_config["calibration_n_bins"] is not int

    Returns:
        Axes: Calibration Curve for Evaluated Classifier
    """

    # check that types are appropriate
    if calibration_config:
        # check that name provided for classifier is of proper type
        if not isinstance(
            calibration_config.get("calibration_classifier_name", ""), str
        ):
            raise TypeError("calibration_classifier_name should be of type string")

        # check that name provided for calibration n_bins is of proper type
        if not isinstance(calibration_config.get("calibration_n_bins", ""), int):
            raise TypeError("calibration_n_bins should be of type int")

    return CalibrationDisplay.from_predictions(
        y_true,
        y_prob=y_probs[:, 1],
        pos_label=pos_label,
        name=calibration_config.get("calibration_classifier_name", None),  # type: ignore
        n_bins=calibration_config.get("calibration_n_bins", 10),  # type: ignore
    ).ax_
