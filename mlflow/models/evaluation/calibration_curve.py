import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.calibration import CalibrationDisplay, calibration_curve


def make_multi_class_calibration_plot(
    n_classes, y_true, y_probs, calibration_config, label_list
) -> Figure:
    """Generate one calibration plot for all classes of a multi-class classifier

    Args:
        n_classes (int): number of classes in multi-class prediction
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probs (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        calibration_config (dict[str, Union[str, int]]):
            Additional parameters for sklearn.CalibrationDisplay
        label_list (array-like): list of label names for each class

    Returns:
        Figure: Multi-Class Calibration Plot
    """
    fig, ax = plt.subplots()
    # add calibration line for each class
    for _class in range(n_classes):
        curves = calibration_curve(
            y_true=[v == _class for v in y_true],
            y_prob=y_probs[:, _class],
            n_bins=calibration_config.get("calibration_n_bins", 10),
        )
        plt.plot(
            curves[0],
            curves[1],
            marker="o",
            markersize=3,
            label=f"Class {label_list[_class]}",
        )
    # plot perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="dotted", label="Perfect Calibration", color="black")
    # add legend to plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=n_classes // 3,
    )
    # set figure title
    ax.set_title(
        f"{calibration_config.get('calibration_classifier_name', 'Classifier')} Calibration",
        fontsize=12,
    )
    # set figure axis labels
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of True Positives", fontsize=10)
    return fig


def plot_calibration_curve(y_true, y_probs, pos_label, calibration_config, label_list) -> Figure:
    """Generate a calibration curve for a trained classifier

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probs (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        pos_label (str): Label for the positive class.

        calibration_config (dict[str, Union[str, int]]):
            Additional parameters for sklearn.CalibrationDisplay

        label_list: List of class names

    Raises:
        TypeError: if calibration_config["calibration_classifier_name"] is not str
        TypeError: if calibration_config["calibration_n_bins"] is not int

    Returns:
        Figure: Calibration Curve for Evaluated Classifier
    """
    # get number of classes
    n_classes = y_probs.shape[-1]

    # check that types are appropriate
    if calibration_config:
        # check that name provided for classifier is of proper type
        if not isinstance(calibration_config.get("calibration_classifier_name", ""), str):
            raise TypeError("calibration_classifier_name should be of type string")

        # check that name provided for calibration n_bins is of proper type
        if not isinstance(calibration_config.get("calibration_n_bins", 10), int):
            raise TypeError("calibration_n_bins should be of type int")

    # if we are evaluating a binary classifier assume positive class is at column index 1
    if n_classes == 2:
        return CalibrationDisplay.from_predictions(
            y_true,
            y_prob=y_probs[:, 1],
            pos_label=pos_label,
            name=calibration_config.get("calibration_classifier_name", None),  # type: ignore
            n_bins=calibration_config.get("calibration_n_bins", 10),  # type: ignore
        ).figure_

    # evaluating a multi-class classifier, create a calibration curve for each class

    return make_multi_class_calibration_plot(
        n_classes, y_true, y_probs, calibration_config, label_list
    )
