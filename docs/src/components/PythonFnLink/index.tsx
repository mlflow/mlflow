import DLink from "@docusaurus/Link";

// Link map for Python function names to their corresponding Docusaurus URLs
// Could be expanded for all links that we use
const linkMap = {
  "mlflow.data.dataset.Dataset":
    "/docs/python_api/mlflow.data#mlflow.data.dataset.Dataset",
  "mlflow.deployments.set_deployments_target":
    "/docs/python_api/mlflow#mlflow.deployments.set_deployments_target",
  "mlflow.evaluate": "/docs/python_api/mlflow#mlflow.evaluate",
  "mlflow.metrics.MetricValue":
    "/docs/python_api/mlflow#mlflow.metrics.MetricValue",
  "mlflow.metrics.flesch_kincaid_grade_level":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.flesch_kincaid_grade_level",
  "mlflow.metrics.genai.EvaluationExample":
    "/docs/python_api/mlflow#mlflow.metrics.genai.EvaluationExample",
  "mlflow.metrics.genai.answer_correctness":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.genai.answer_correctness",
  "mlflow.metrics.genai.answer_relevance":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.genai.answer_relevance",
  "mlflow.metrics.genai.answer_similarity":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.genai.answer_similarity",
  "mlflow.metrics.genai.faithfulness":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.genai.faithfulness",
  "mlflow.metrics.genai.make_genai_metric":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.genai.make_genai_metric",
  "mlflow.metrics.genai.relevance":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.genai.relevance",
  "mlflow.metrics.latency":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.latency",
  "mlflow.metrics.rougeL":
    "/docs/python_api/mlflow.metrics#mlflow.metrics.rougeL",
  "mlflow.models.EvaluationResult":
    "/docs/python_api/mlflow#mlflow.models.EvaluationResult",
  "mlflow.openai.log_model": "/docs/python_api/mlflow#mlflow.openai.log_model",
  "mlflow.pyfunc.PyFuncModel":
    "/docs/python_api/mlflow#mlflow.pyfunc.PyFuncModel",
  "mlflow.pyfunc.PyFuncModel.predict":
    "/docs/python_api/mlflow#mlflow.pyfunc.PyFuncModel.predict",
};

interface LinkProps {
  fn: keyof typeof linkMap;
  code: boolean;
}

const PythonFnLink = ({ fn, code = true }: LinkProps): JSX.Element => (
  <DLink href={linkMap[fn]}>{code ? <code>{fn}()</code> : fn + "()"}</DLink>
);

export default PythonFnLink;
