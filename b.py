# with a model and using `evaluator_config`
mlflow.evaluate(
    model=retriever_function,
    data=data,
    targets="ground_truth",
    model_type="retriever",
    evaluators="default",
    evaluator_config={"retriever_k": 5}
)
