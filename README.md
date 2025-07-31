<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>
<h2 align="center" style="border-bottom: none">Open-Source Platform for Productionizing AI</h2>

MLflow is an open-source developer platform to build AI/LLM applications and models with confidence. Enhance your AI applications with end-to-end **experiment tracking**, **observability**, and **evaluations**, all in one integrated platform.

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlflow)](https://pypi.org/project/mlflow/)
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
      <a href="https://mlflow.org/" style="font-size: 1.2em;"><strong>Website</strong></a> ·
      <a href="https://mlflow.org/docs/latest/index.html" style="font-size: 1.2em;"><strong>Docs</strong></a> ·
      <a href="https://github.com/mlflow/mlflow/issues/new/choose" style="font-size: 1.2em;"><strong>Feature Request</strong></a> ·
      <a href="https://mlflow.org/blog" style="font-size: 1.2em;"><strong>News</strong></a> ·
      <a href="https://www.youtube.com/@mlflowoss" style="font-size: 1.2em;"><strong>YouTube</strong></a> ·
      <a href="https://lu.ma/mlflow?k=c" style="font-size: 1.2em;"><strong>Events</strong></a>
   </div>
</div>

## 📦 Core Components

MLflow is an **only platform that provides a unified solution for all your AI/ML needs**, including LLMs, Agents, Deep Learning, and traditional machine learning.

### 💡 For LLM / GenAI Developers

<div style="display: flex; flex-direction: row; gap: 20px;">
  <div>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-tracing.png" alt="Tracing" width=100%>
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/llms/tracing/index.html">🔍 Tracing / Observability</a>
        <br>
        <div style="margin-top: 10px;">Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.</div>
    </div>
  </div>
  <div>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-llm-eval.png" alt="LLM Evaluation" width=100%>
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/genai/eval-monitor/">📊 LLM Evaluation</a>
        <br>
        <div style="margin-top: 10px;">A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.</div>
    </div>
  </div>
</div>

<br>
<div style="display: flex; flex-direction: row; gap: 20px;">
  <div>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-prompt.png" alt="Prompt Management" width=95% style="margin-left: 20px;">
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/">🤖 Prompt Management</a>
        <br>
        <div style="margin-top: 10px;">Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.</div>
    </div>
  </div>
  <div>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-logged-model.png" alt="MLflow Hero" width=95% style="margin-left: 20px;">
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/">📦 App Version Tracking</a>
        <br>
        <div style="margin-top: 10px;">MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.</div>
    </div>
  </div>
</div>

### 🎓 For Data Scientists

<div style="text-align: center; width: 100%;">
  <div style="width: 60%; margin: 0 auto;">
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-traditional-ml.png" alt="Tracking" width=100%>
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/ml/tracking/"> 📝 Experiment Tracking</a>
        <br>
        <div style="margin-top: 10px;">Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.</div>
    </div>
  </div>
</div>

<div style="display: flex; flex-direction: row; gap: 20px;">
  <div>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-model-registry.png" alt="Model Registry" width=100%>
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/ml/model-registry/"> 💾 Model Registry</a>
        <br>
        <div style="margin-top: 10px;"> A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.</div>
    </div>
  </div>
  <div>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-deployment.png" alt="Deployment" width=100%>
    <div style="text-align: center; ">
        <a href="https://mlflow.org/docs/latest/ml/deployment/"> 🚀 Deployment</a>
        <br>
        <div style="margin-top: 10px;"> Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.</div>
    </div>
  </div>
</div>

## 🚀 Installation

To install the MLflow Python package, run the following command:

```
pip install mlflow
```

## 🌐 Hosting MLflow Anywhere

<div align="center" >
  <img src="./assets/readme-providers.png" alt="Providers" width=100%>
</div>

You can run MLflow in many different environments, including local machines, on-premise servers, and cloud infrastructure.

Trusted by thousands of organizations, MLflow is now offered as a managed service by most major cloud providers:

- [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
- [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
- [Databricks](https://www.databricks.com/product/managed-mlflow)
- [Nebius](https://nebius.com/services/managed-mlflow)

For hosting MLflow on your own infrastructure, please refer to [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Multi-Language Support

- [Python](https://pypi.org/project/mlflow/)
- [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
- [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
- [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## 🔗 Integrations

MLflow is natively integrated with many popular machine learning frameworks and GenAI libraries. Click on each logo to see the detailed documentation and examples for each integration.

<div align="center" style="padding: 20px;">
<table style="border: 1px solid #ddd; border-collapse: collapse; background-color: white; width: 100%; table-layout: fixed;">
  <tr>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle; height: 80px;">
      <a href="https://mlflow.org/docs/latest/ml/traditional-ml/sklearn/index.html" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/scikit-learn-logo.svg" width="80"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/traditional-ml/xgboost/index.html" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/xgboost-logo.svg" width="80"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/model/#lightgbm-lightgbm" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/lightgbm-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/model/index.html#catboost-catboost" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/catboost-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding: 10px; max-width: 16%; text-align: center; vertical-align: middle; height: 60px;">
      <a href="https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/intro/optuna-logo.jpeg"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/model/index.html#statsmodels-statsmodels" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/statsmodels-logo.svg"/>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 10px; max-width: 16%; text-align: center; vertical-align: middle; height: 80px;">
      <a href="https://mlflow.org/docs/latest/ml/deep-learning/pytorch/index.html" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/pytorch-logo.svg"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/deep-learning/transformers/index.html" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/huggingface-logo.svg" style="height: 40px;"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/deep-learning/keras/index.html" target="_blank">
      <div style="display: flex; align-items: center; justify-content: center;">
          <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/keras-logo.svg" width="30"/> <div style="margin-left: 8px; color: #666;">Keras</div>
        </div>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/deep-learning/tensorflow/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/TensorFlow-logo.svg"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/deep-learning/sentence-transformers/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/sentence-transformers-logo.png" width="120"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/ml/deep-learning/spacy/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/spacy-logo.svg" width="80"/>
    </td>
  </tr>
  <tr >
    <td style="border: 1px solid #ddd; padding: 10px; max-width: 16%; text-align: center; vertical-align: middle; height: 80px;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/openai/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/openai-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langchain/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/langchain-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/langgraph-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/llama_index/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/llamaindex-logo.svg"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/anthropic/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/anthropic-logo.svg"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/flavors/dspy/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/dspy-logo.png" width="80"/>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 10px; max-width: 16%; text-align: center; vertical-align: middle; height: 80px;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/bedrock/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/bedrock-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/autogen/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/autogen-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/semantic_kernel/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/semantic-kernel-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pydantic_ai/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/pydanticai-logo.png" />
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/google-gemini-logo.svg" width="80"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/litellm/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/litellm-logo.jpg"/>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 10px; max-width: 16%; text-align: center; vertical-align: middle; height: 80px;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/crewai/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/crewai-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/ollama/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/ollama-logo.png" width="50"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/smolagents/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/smolagents-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/deepseek/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/deepseek-logo.png"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/groq/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/groq-logo.svg" width="80"/>
    </td>
    <td style="border: 1px solid #ddd; padding:10px; max-width: 16%; text-align: center; vertical-align: middle;">
      <a href="https://mlflow.org/docs/latest/genai/tracing/integrations/listing/mistral/" target="_blank">
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/docs/static/images/logos/mistral-ai-logo.svg"/>
    </td>
  </tr>
</table>
</div>

## 💭 Support

- For help or questions about MLflow usage (e.g. "how do I do X?") visit the [documentation](https://mlflow.org/docs/latest/index.html).
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
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" style="border-radius: 15px;" />
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
- [Weichen Xu](https://github.com/WeichenXu123)
- [Yuki Watanabe](https://github.com/B-Step62)
- [Tomu Hirata](https://github.com/TomeHirata)

<style>
@media (max-width: 768px) {
  div[style*="flex-direction: row"] {
    flex-direction: column !important;
  }
}
</style>