# MLflow examples for LLM use cases

This directory includes several examples for tracking, evaluating, and scoring models with LLMs.

## Summarization
The ``summarization/summarization.py`` script uses prompt engineering to build a summarization model for news articles with LangChain. It leverages the ``mlflow.langchain`` flavor to package and log the model to MLflow, ``mlflow.evaluate()`` to evaluate the model's performance on a small example dataset, and ``mlflow.pyfunc.load_model()`` to load and score the packaged model on a new example article.

To run the example, simply execute the following command from this directory:

```
%sh python summarization/summarization.py
```

You must have [LangChain](https://python.langchain.com/en/latest/index.html) installed in order to run the example, and we recommend installing the [Hugging Face Evaluate library](https://huggingface.co/docs/evaluate/index) in order to compute [ROUGE metrics](https://en.wikipedia.org/wiki/ROUGE_(metric) for summary quality. Additionally, you must specify a valid OpenAI API key in the ``OPENAI_API_KEY`` environment variable.

## Question answering
The ``question_answering.py`` script uses prompt engineering to build a model that answers questions about MLflow.

It leverages the ``mlflow.openai`` flavor to package and log the model to MLflow, ``mlflow.evaluate()`` to evaluate the model's performance on some example questions, and ``mlflow.pyfunc.load_model()`` to load and score the packaged model on a new example question.

To run the example, simply execute the following command from this directory:

```
%sh python question_answering/question_answering.py
```

You must have the [OpenAI Python client](https://pypi.org/project/openai/) installed in order to run the example. Additionally, you must specify a valid OpenAI API key in the ``OPENAI_API_KEY`` environment variable.
