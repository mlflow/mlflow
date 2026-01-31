# ruff: noqa
# clint: skip-file
"""
Script to run LLM calls over AIME dataset using a registered prompt for distillation tracing.
Traces are automatically captured and linked to the prompt.
"""

import mlflow
import openai
from datasets import load_dataset


def create_aime_dataset(experiment_id: str, num_samples: int = 50) -> str:
    """Create AIME evaluation dataset with specified number of samples."""
    from mlflow.genai.datasets import create_dataset

    dataset_name = f"aime_distillation_{num_samples}"

    print("Loading AIME dataset from HuggingFace...")
    hf_dataset = load_dataset("gneubig/aime-1983-2024", split="train")

    records = [
        {
            "inputs": {"question": item["Question"]},
            "expectations": {"expected_response": item["Answer"]},
        }
        for item in hf_dataset
    ][:num_samples]

    print(f"Creating dataset '{dataset_name}' with {len(records)} records...")
    dataset = create_dataset(name=dataset_name, experiment_id=experiment_id)
    dataset.merge_records(records)

    print(f"Dataset created: {dataset.dataset_id}")
    return dataset.dataset_id


# Define the traced function - load_prompt MUST be called inside a traced context
# for the prompt to be linked to the trace
@mlflow.trace
def solve_aime_problem(client, question: str, prompt_uri: str, model_name: str, temperature: float):
    """Solve an AIME problem using the prompt. Prompt is loaded inside trace for linking."""
    # Load prompt inside the trace - this links the prompt to this trace
    prompt = mlflow.genai.load_prompt(prompt_uri)

    formatted = prompt.format(question=question)
    messages = [{"role": "user", "content": formatted}] if prompt.is_text_prompt else formatted

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("optimization_backend")

    # Create dataset
    # create_aime_dataset(experiment.experiment_id, num_samples=50)

    # Load prompt once to get config (not for linking)
    prompt = mlflow.genai.load_prompt("prompts:/aime_solver/9")
    print(f"Loaded prompt: {prompt.name} v{prompt.version}")
    print(f"Model config: {prompt.model_config}")

    config = prompt.model_config
    model_name = config.get("model_name")
    temperature = config.get("temperature", 1)
    prompt_uri = "prompts:/aime_solver/9"

    # Enable tracing
    mlflow.openai.autolog()

    # Load questions
    hf_dataset = load_dataset("gneubig/aime-1983-2024", split="train")
    questions = [item["Question"] for item in hf_dataset][:50]

    # Run LLM calls - prompt is loaded inside traced function for proper linking
    client = openai.OpenAI()
    for i, question in enumerate(questions):
        print(f"Processing {i + 1}/50...")
        result = solve_aime_problem(client, question, prompt_uri, model_name, temperature)
        print(f"  Response length: {len(result)} chars")

    print(f"\nDone! View traces at: http://127.0.0.1:5000/#/experiments/{experiment.experiment_id}")


if __name__ == "__main__":
    main()
