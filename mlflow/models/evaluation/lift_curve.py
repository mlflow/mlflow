import matplotlib.pyplot as plt
import numpy as np


def _cumulative_gain_curve(y_true, y_score, pos_label=None):
    """
    This method is copied from scikit-plot package.
    See https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/helpers.py#L157

    This function generates the points necessary to plot the Cumulative Gain

    Note: This implementation is restricted to the binary classification task.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).

        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative

    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.

        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


def plot_lift_curve(
    y_true,
    y_probas,
    title="Lift Curve",
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    pos_label=None,
):
    """
    This method is copied from scikit-plot package.
    See https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/metrics.py#L1133

    Generates the Lift Curve from labels and scores/probabilities

    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html.
    The implementation here works only for binary classification.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Lift Curve".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

        pos_label (optional): Label for the positive class.

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> plot_lift_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_lift_curve.png
           :align: center
           :alt: Lift Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(f"Cannot calculate Lift Curve for data with {len(classes)} category/ies")

    # Compute Cumulative Gain Curves
    percentages, gains1 = _cumulative_gain_curve(y_true, y_probas[:, 0], classes[0])
    percentages, gains2 = _cumulative_gain_curve(y_true, y_probas[:, 1], classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains2 = gains2[1:]

    gains1 = gains1 / percentages
    gains2 = gains2 / percentages

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    label0 = f"Class {classes[0]}"
    label1 = f"Class {classes[1]}"
    # show (positive) next to the positive class in the legend
    if pos_label:
        if pos_label == classes[0]:
            label0 = f"Class {classes[0]} (positive)"
        elif pos_label == classes[1]:
            label1 = f"Class {classes[1]} (positive)"
        # do not mark positive class if pos_label is not in classes

    ax.plot(percentages, gains1, lw=3, label=label0)
    ax.plot(percentages, gains2, lw=3, label=label1)

    ax.plot([0, 1], [1, 1], "k--", lw=2, label="Baseline")

    ax.set_xlabel("Percentage of sample", fontsize=text_fontsize)
    ax.set_ylabel("Lift", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid("on")
    ax.legend(loc="best", fontsize=text_fontsize)

    return ax
