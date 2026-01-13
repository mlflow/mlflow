from datasets import load_dataset


def create_aime_evaluation_dataset(experiment_id: str) -> str:
    from mlflow.genai.datasets import create_dataset

    mlflow_dataset_name = "aime_1983_2024_tailored"

    print("Loading AIME dataset from HuggingFace...")
    hf_dataset = load_dataset("gneubig/aime-1983-2024", split="train")

    print(f"Loaded {len(hf_dataset)} samples")

    # Transform to MLflow format
    # - inputs: the question/problem to solve
    # - expectations: the expected answer (ground truth)
    records = []
    for item in hf_dataset:
        record = {
            "inputs": {
                "question": item["Question"],
            },
            "expectations": {
                "expected_response": item["Answer"],
            },
        }
        records.append(record)

    records = records[:20]

    print(f"Transformed {len(records)} records to MLflow format")

    # Create the dataset
    dataset_tags = {
        "source": "huggingface",
        "hf_dataset": "gneubig/aime-1983-2024",
        "purpose": "prompt_optimization",
    }

    print(f"Creating MLflow EvaluationDataset '{mlflow_dataset_name}'...")
    dataset = create_dataset(
        name=mlflow_dataset_name,
        experiment_id=experiment_id,
        tags=dataset_tags,
    )

    print(f"Created dataset with ID: {dataset.dataset_id}")

    # Add records to the dataset
    print(f"Merging {len(records)} records into dataset...")
    dataset.merge_records(records)

    print("Dataset created successfully!")
    print(f"  - Name: {dataset.name}")
    print(f"  - ID: {dataset.dataset_id}")
    print(f"  - Records: {len(records)}")

    return dataset.dataset_id


def main():
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("optimization_backend")
    experiment_id = experiment.experiment_id

    dataset_id = create_aime_evaluation_dataset(experiment_id)
    print(f"Dataset created with ID: {dataset_id}")


if __name__ == "__main__":
    main()
