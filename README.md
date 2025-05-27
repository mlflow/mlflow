# MLflow: A Machine Learning Lifecycle Platform

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible

---

The core components of MLflow are:

- [Experiment Tracking](https://mlflow.org/docs/latest/tracking.html) 📝: A set of APIs to log models, params, and results in ML experiments and compare them using an interactive UI.
- [Model Packaging](https://mlflow.org/docs/latest/models.html) 📦: A standard format for packaging a model and its metadata, such as dependency versions, ensuring reliable deployment and strong reproducibility.
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html) 💾: A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models.
- [Serving](https://mlflow.org/docs/latest/deployment/index.html) 🚀: Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
- [Evaluation](https://mlflow.org/docs/latest/model-evaluation/index.html) 📊: A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to record model performance and visually compare results across multiple models.
- [Observability](https://mlflow.org/docs/latest/llms/tracing/index.html) 🔍: Tracing integrations with various GenAI libraries and a Python SDK for manual instrumentation, offering smoother debugging experience and supporting online monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

To install the MLflow Python package, run the following command:

```
pip install mlflow
```

Alternatively, you can install MLflow from on different package hosting platforms:

|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation 📘

Official documentation for MLflow can be found at [here](https://mlflow.org/docs/latest/index.html).

## Running Anywhere 🌐

You can run MLflow on many different environments, including local development, Amazon SageMaker, AzureML, and Databricks. Please refer to [this guidance](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for how to setup MLflow on your environment.

## Usage

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/tracking.html))

The following examples trains a simple regression model with scikit-learn, while enabling MLflow's [autologging](https://mlflow.org/docs/latest/tracking/autolog.html) feature for experiment tracking.

```python
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Enable MLflow's automatic experiment tracking for scikit-learn
mlflow.sklearn.autolog()

# Load the training dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# MLflow triggers logging automatically upon model fitting
rf.fit(X_train, y_train)
```

Once the above code finishes, run the following command in a separate terminal and access the MLflow UI via the printed URL. An MLflow **Run** should be automatically created, which tracks the training dataset, hyper parameters, performance metrics, the trained model, dependencies, and even more.

```
mlflow ui
```

### Serving Models ([Doc](https://mlflow.org/docs/latest/deployment/index.html))

You can deploy the logged model to a local inference server by a one-line command using the MLflow CLI. Visit the documentation for how to deploy models to other hosting platforms.

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

The following example runs automatic evaluation for question-answering tasks with several built-in metrics.

```python
import mlflow
import pandas as pd

# Evaluation set contains (1) input question (2) model outputs (3) ground truth
df = pd.DataFrame(
    {
        "inputs": ["What is MLflow?", "What is Spark?"],
        "outputs": [
            "MLflow is an innovative fully self-driving airship powered by AI.",
            "Sparks is an American pop and rock duo formed in Los Angeles.",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics.",
        ],
    }
)
eval_dataset = mlflow.data.from_pandas(
    df, predictions="outputs", targets="ground_truth"
)

# Start an MLflow Run to record the evaluation results to
with mlflow.start_run(run_name="evaluate_qa"):
    # Run automatic evaluation with a set of built-in metrics for question-answering models
    results = mlflow.evaluate(
        data=eval_dataset,
        model_type="question-answering",
    )

print(results.tables["eval_results_table"])
```

### Observability ([Doc](https://mlflow.org/docs/latest/llms/tracing/index.html))

MLflow Tracing provides LLM observability for various GenAI libraries such as OpenAI, LangChain, LlamaIndex, DSPy, AutoGen, and more. To enable auto-tracing, call `mlflow.xyz.autolog()` before running your models. Refer to the documentation for customization and manual instrumentation.

```python
import mlflow
from openai import OpenAI

# Enable tracing for OpenAI
mlflow.openai.autolog()

# Query OpenAI LLM normally
response = OpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi!"}],
    temperature=0.1,
)
```

Then navigate to the "Traces" tab in the MLflow UI to find the trace records OpenAI query.

## Community

- For help or questions about MLflow usage (e.g. "how do I do X?") visit the [docs](https://mlflow.org/docs/latest/index.html)
  or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
- Alternatively, you can ask the question to our AI-powered chat bot. Visit the doc website and click on the **"Ask AI"** button at the right bottom to start chatting with the bot.
- To report a bug, file a documentation issue, or submit a feature request, please [open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
- For release announcements and other discussions, please subscribe to our mailing list (mlflow-users@googlegroups.com)
  or join us on [Slack](https://mlflow.org/slack).

## Contributing

We happily welcome contributions to MLflow! We are also seeking contributions to items on the
[MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3). Please see our
[contribution guide](CONTRIBUTING.md) to learn more about contributing to MLflow.

## Core Members

MLflow is currently maintained by the following core members with significant contributions from hundreds of exceptionally talented community members.

- [Ben Wilson](https://github.com/BenWilson2)
- [Corey Zumar](https://github.com/dbczumar)
- [Daniel Lok](https://github.com/daniellok-db)
- [Gabriel Fu](https://github.com/gabrielfu)
- [Harutaka Kawamura](https://github.com/harupy)
- [Serena Ruan](https://github.com/serena-ruan)
- [Weichen Xu](https://github.com/WeichenXu123)
- [Yuki Watanabe](https://github.com/B-Step62)
- [Tomu Hirata](https://github.com/TomeHirata)
