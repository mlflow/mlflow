<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>
<h2 align="center" style="border-bottom: none">Open-Source Platform for Productionizing AI</h2>

MLflow is an open-source developer platform to build AI/LLM applications and models with confidence. Enhance your AI applications with end-to-end **experiment tracking**, **observability**, and **evaluations**, all in one integrated platform.

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlflow)](https://pepy.tech/projects/mlflow)
[![License](https://img.shields.io/github/license/mlflow/mlflow)](https://github.com/mlflow/mlflow/blob/main/LICENSE)
<a href="https://twitter.com/intent/follow?screen_name=mlflow" target="_blank">
<img src="https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5"
      alt="follow on X(Twitter)"></a>
<a href="https://www.linkedin.com/company/mlflow-org/" target="_blank">
<img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff"
      alt="follow on LinkedIn"></a>
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

</div>

<div align="center">
   <div>
      <a href="https://mlflow.org/"><strong>Website</strong></a> ·
      <a href="https://mlflow.org/docs/latest"><strong>Docs</strong></a> ·
      <a href="https://github.com/mlflow/mlflow/issues/new/choose"><strong>Feature Request</strong></a> ·
      <a href="https://mlflow.org/blog"><strong>News</strong></a> ·
      <a href="https://www.youtube.com/@mlflowoss"><strong>YouTube</strong></a> ·
      <a href="https://lu.ma/mlflow?k=c"><strong>Events</strong></a>
   </div>
</div>

<br>

## 🚀 Installation

To install the MLflow Python package, run the following command:

```
pip install mlflow
```

## 📦 Core Components

MLflow is **the only platform that provides a unified solution for all your AI/ML needs**, including LLMs, Agents, Deep Learning, and traditional machine learning.

### 💡 For LLM / GenAI Developers

<table>
  <tr>
    <td>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-tracing.png" alt="Tracing" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/llms/tracing/index.html"><strong>🔍 Tracing / Observability</strong></a>
        <br><br>
        <div>Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/tracing/quickstart/python-openai/">Getting Started →</a>
        <br><br>
    </div>
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-llm-eval.png" alt="LLM Evaluation" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/genai/eval-monitor/"><strong>📊 LLM Evaluation</strong></a>
        <br><br>
        <div>A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/eval-monitor/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-prompt.png" alt="Prompt Management">
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/"><strong>🤖 Prompt Management</strong></a>
        <br><br>
        <div>Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/">Getting Started →</a>
        <br><br>
    </div>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-logged-model.png" alt="MLflow Hero">
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/"><strong>📦 App Version Tracking</strong></a>
        <br><br>
        <div>MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/version-tracking/quickstart/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
</table>

### 🎓 For Data Scientists

<table>
  <tr>
    <td colspan="2" align="center" >
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-experiment.png" alt="Tracking" width=50%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/ml/tracking/"><strong>📝 Experiment Tracking</strong></a>
        <br><br>
        <div>Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.</div><br>
        <a href="https://mlflow.org/docs/latest/ml/tracking/quickstart/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-model-registry.png" alt="Model Registry" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/ml/model-registry/"><strong>💾 Model Registry</strong></a>
        <br><br>
        <div> A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.</div><br>
        <a href="https://mlflow.org/docs/latest/ml/model-registry/tutorial/">Getting Started →</a>
        <br><br>
    </div>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-deployment.png" alt="Deployment" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/ml/deployment/"><strong>🚀 Deployment</strong></a>
        <br><br>
        <div> Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.</div><br>
        <a href="https://mlflow.org/docs/latest/ml/deployment/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
</table>

## 🌐 Hosting MLflow Anywhere

<div align="center" >
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-providers.png" alt="Providers" width=100%>
</div>

You can run MLflow in many different environments, including local machines, on-premise servers, and cloud infrastructure.

Trusted by thousands of organizations, MLflow is now offered as a managed service by most major cloud providers:

- [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
- [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
- [Databricks](https://www.databricks.com/product/managed-mlflow)
- [Nebius](https://nebius.com/services/managed-mlflow)

For hosting MLflow on your own infrastructure, please refer to [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Supported Programming Languages

- [Python](https://pypi.org/project/mlflow/)
- [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
- [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
- [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## 🔗 Integrations

MLflow is natively integrated with many popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/ml/tracking/))

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
            "MLflow is an open-source platform for productionizing AI.",
            "Apache Spark is an open-source, distributed computing system.",
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

## 💭 Support

- For help or questions about MLflow usage (e.g. "how do I do X?") visit the [documentation](https://mlflow.org/docs/latest).
- In the documentation, you can ask the question to our AI-powered chat bot. Click on the **"Ask AI"** button at the right bottom.
- Join the [virtual events](https://lu.ma/mlflow?k=c) like office hours and meetups.
- To report a bug, file a documentation issue, or submit a feature request, please [open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
- For release announcements and other discussions, please subscribe to our mailing list (mlflow-users@googlegroups.com)
  or join us on [Slack](https://mlflow.org/slack).

## 🤝 Contributing

We happily welcome contributions to MLflow!

- Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
- Contribute for [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
- Writing about MLflow and sharing your experience

Please see our [contribution guide](CONTRIBUTING.md) to learn more about contributing to MLflow.

## ⭐️ Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## ✏️ Citation

If you use MLflow in your research, please cite it using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow), which will provide you with citation formats including APA and BibTeX.

## 👥 Core Members

MLflow is currently maintained by the following core members with significant contributions from hundreds of exceptionally talented community members.

- [Ben Wilson](https://github.com/BenWilson2)
- [Corey Zumar](https://github.com/dbczumar)
- [Daniel Lok](https://github.com/daniellok-db)
- [Gabriel Fu](https://github.com/gabrielfu)
- [Harutaka Kawamura](https://github.com/harupy)
- [Serena Ruan](https://github.com/serena-ruan)
- [Tomu Hirata](https://github.com/TomeHirata)
- [Weichen Xu](https://github.com/WeichenXu123)
- [Yuki Watanabe](https://github.com/B-Step62)
