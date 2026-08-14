# MLflow examples for LLM use cases

This directory includes several examples for tracking, evaluating, and scoring models with LLMs.

## Summarization

The `summarization/summarization.py` script uses prompt engineering to build two summarization models for news articles with LangChain. It leverages the `mlflow.langchain` flavor to package and log the models to MLflow, `mlflow.evaluate()` to evaluate each model's performance on a small example dataset, and `mlflow.pyfunc.load_model()` to load and score the best packaged model on a new example article.

To run the example as an MLflow Project, simply execute the following command from this directory:

```
$ cd summarization && mlflow run .
```

To run the example as a Python script, simply execute the following command from this directory:

```
$ cd summarization && python summarization.py
```

Note that this example requires MLflow 2.4.0 or greater to run. Additionally, you must have [LangChain](https://python.langchain.com/en/latest/index.html) and the [OpenAI Python client](https://pypi.org/project/openai/) installed in order to run the example. We also recommend installing the [Hugging Face Evaluate library](https://huggingface.co/docs/evaluate/index) to compute [ROUGE metrics](<https://en.wikipedia.org/wiki/ROUGE_(metric)>) for summary quality. Finally, you must specify a valid OpenAI API key in the `OPENAI_API_KEY` environment variable.

## Question answering

The `question_answering/question_answering.py` script uses prompt engineering to build two models that answer questions about MLflow.

It leverages the `mlflow.openai` flavor to package and log the models to MLflow, `mlflow.evaluate()` to evaluate each model's performance on some example questions, and `mlflow.pyfunc.load_model()` to load and score the best packaged model on a new example question.

To run the example as an MLflow Project, simply execute the following command from this directory:

```
$ cd question_answering && mlflow run .
```

To run the example as a Python script, simply execute the following command from this directory:

```
$ cd question_answering && python question_answering.py
```

Note that this example requires MLflow 2.4.0 or greater to run. Additionally, you must have the [OpenAI Python client](https://pypi.org/project/openai/), [tiktoken](https://pypi.org/project/tiktoken/), and [tenacity](https://pypi.org/project/tenacity/) installed in order to run the example. Finally, you must specify a valid OpenAI API key in the `OPENAI_API_KEY` environment variable.
