from sklearn.metrics import mean_squared_error


def weighted_mean_squared_error(eval_df, _builtin_metrics):
    return mean_squared_error(
        eval_df["prediction"],
        eval_df["target"],
        sample_weight=1 / eval_df["prediction"].values,
    )
