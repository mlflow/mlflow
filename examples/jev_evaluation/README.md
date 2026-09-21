# Evaluate support answers with Jev

This example evaluates two precomputed support answers using MLflow's native
`make_jev_scorer` API. It demonstrates all three answer types:

- `answer_relevance`: a `noul` probability converted to a boolean with a 0.7 threshold.
- `request_category`: a `choice` label.
- `answer_completeness`: a fractional `score` over three ordered rubric levels.

MLflow records the original probabilities in each assessment's metadata, including
when a threshold produces a boolean. Jev does not produce a text rationale.
Each scorer sends one question, with `inputs`, `outputs`, and `expectations` as
structured state.

## Setup

Install MLflow with GenAI dependencies and start the tracking server in a separate
terminal:

```bash
pip install 'mlflow[genai]'
mlflow server --port 5000
```

From this directory, configure the tracking server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Run with a local TypeSafe API key

Set `TYPESAFE_API_KEY` to your TypeSafe API key in your shell, then run:

```bash
python evaluate.py
```

The default model is `typesafe:/jev-latest`. This flow calls TypeSafe directly
from the Python process and makes six provider requests: two examples times three
scorers. Calls use your TypeSafe account's quota.

## Run through AI Gateway

In the MLflow UI:

1. Open **AI Gateway** and click **Create Endpoint**.
2. Name the endpoint `jev-evaluator` and choose **TypeSafe** and `jev-latest`.
3. Create or select a TypeSafe LLM connection containing your API key.
4. Create the endpoint.

Run the same evaluation through that endpoint:

```bash
python evaluate.py --model gateway:/jev-evaluator
```

This flow uses the API key stored in the gateway. The client only needs access to
the MLflow server. If your server uses authentication, configure your MLflow
client credentials as usual.

Open the `jev-evaluation` experiment to inspect the evaluation run and its trace
assessments. The [Jev scorer guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/custom-judges/jev/)
shows how to create the same scorer in the UI, register versions, and enable
automatic evaluation. Saved scorers and UI execution require a gateway endpoint.
