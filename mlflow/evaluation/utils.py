from typing import List, Tuple

import pandas as pd

from mlflow.entities.evaluation import Evaluation
from mlflow.entities.feedback import Feedback
from mlflow.entities.metric import Metric


def evaluations_to_dataframes(
    evaluations: List[Evaluation],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Converts a list of Evaluation objects to three separate DataFrames: one for main evaluation
    data (excluding feedback and metrics), one for metrics, and one for feedback.

    Args:
        evaluations (List[Evaluation]): List of Evaluation objects.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
            1. DataFrame with the main evaluation data (excluding feedback and metrics).
            2. DataFrame with metrics.
            3. DataFrame with feedback.
    """
    main_data = []
    metrics_data = []
    feedback_data = []

    for evaluation in evaluations:
        eval_dict = evaluation.to_dictionary()

        # Extract feedback and metrics
        feedback_list = eval_dict.pop("feedback", [])
        metrics_list = eval_dict.pop("metrics", [])

        main_data.append(eval_dict)

        for metric_dict in metrics_list:
            metric_dict["evaluation_id"] = eval_dict["evaluation_id"]
            metrics_data.append(metric_dict)

        for fb_dict in feedback_list:
            fb_dict["evaluation_id"] = eval_dict["evaluation_id"]
            feedback_data.append(fb_dict)

    main_df = pd.DataFrame(main_data)
    metrics_df = pd.DataFrame(metrics_data)
    feedback_df = pd.DataFrame(feedback_data)

    return main_df, metrics_df, feedback_df


def dataframes_to_evaluations(
    main_df: pd.DataFrame, metrics_df: pd.DataFrame, feedback_df: pd.DataFrame
) -> List[Evaluation]:
    """
    Converts three separate DataFrames (main evaluation data, metrics, and feedback) back to a
    list of Evaluation objects.

    Args:
        main_df (pd.DataFrame): DataFrame with the main evaluation data
            (excluding feedback and metrics).
        metrics_df (pd.DataFrame): DataFrame with metrics.
        feedback_df (pd.DataFrame): DataFrame with feedback.

    Returns:
        List[Evaluation]: A list of Evaluation objects created from the DataFrames.
    """

    def group_by_evaluation_id(df: pd.DataFrame, entity_cls):
        """
        Groups rows by 'evaluation_id' and converts them into entities using the provided class.

        Args:
            df (pd.DataFrame): DataFrame to group.
            entity_cls: Class used to create entities from dictionary rows.

        Returns:
            Dict[str, List]: A dictionary with 'evaluation_id' as keys and lists of entity
                instances as values.
        """
        grouped = df.groupby("evaluation_id").apply(
            lambda x: [entity_cls.from_dictionary(row) for row in x.to_dict(orient="records")]
        )
        return grouped.to_dict()

    # Group metrics and feedback by evaluation_id
    metrics_by_eval = group_by_evaluation_id(metrics_df, Metric)
    feedback_by_eval = group_by_evaluation_id(feedback_df, Feedback)

    # Convert main DataFrame to list of dictionaries and create Evaluation objects
    evaluations = []
    for eval_dict in main_df.to_dict(orient="records"):
        evaluation_id = eval_dict["evaluation_id"]
        eval_dict["metrics"] = metrics_by_eval.get(evaluation_id, [])
        eval_dict["feedback"] = feedback_by_eval.get(evaluation_id, [])
        evaluations.append(Evaluation.from_dictionary(eval_dict))

    return evaluations
