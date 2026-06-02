import { Button, ChartLineIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet, SnippetCopyAction } from '@mlflow/mlflow/src/shared/web-shared/snippet';
import { AggregationType, MetricViewType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useTraceMetricsQuery } from '../experiment-overview/hooks/useTraceMetricsQuery';

const getDatasetCodeSnippet = (experimentId: string, scorersDocLink?: string) => `import mlflow
import os
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Define evaluation dataset
eval_dataset = [{
  "inputs": {
    "query": "What is MLflow?",
  }
}]

# Step 2: Define predict_fn
# predict_fn will be called for every row in your evaluation
# dataset. Replace with your app's prediction function.
# NOTE: The **kwargs to predict_fn are the same as the keys of
# the \`inputs\` in your dataset.
def predict(query):
  return query + " an answer"

# Step 3: Run evaluation
# Select scorers relevant to your use case.${scorersDocLink ? `\n# See all available scorers: ${scorersDocLink}` : ''}
evaluate(
  data=eval_dataset,
  predict_fn=predict,
  scorers=[
    Safety(),
    RelevanceToQuery(),
    Guidelines(name="conciseness", guidelines="Responses must be concise."),
  ],
)

# Results will appear back in this UI`;

const getTraceCodeSnippet = (experimentId: string) => `import mlflow
import os
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your API key
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Pull traces to evaluate.
# Adjust max_results, or add a filter_string for time/status, etc.
# See: https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/
traces = mlflow.search_traces(max_results=20)

# Step 2: Run evaluation. No predict_fn needed — inputs/outputs
# are extracted from the trace objects automatically.
evaluate(
  data=traces,
  scorers=[
    Safety(),
    RelevanceToQuery(),
    Guidelines(name="conciseness", guidelines="Responses must be concise."),
  ],
)

# Results will appear back in this UI`;

export const RunEvaluationButton = ({ experimentId }: { experimentId: string }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);
  const evalInstructions = (
    <FormattedMessage
      defaultMessage="Run the following code to start an evaluation."
      description="Instructions for running the evaluation code in OSS"
    />
  );
  const { data: traceMetrics } = useTraceMetricsQuery({
    experimentIds: [experimentId],
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    enabled: isOpen,
  });
  const traceCount = Number(traceMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT] ?? 0);
  const hasTraces = traceCount > 0;
  const codeSnippet = hasTraces ? getTraceCodeSnippet(experimentId) : getDatasetCodeSnippet(experimentId);
  const evalCodeSnippet = (
    <div css={{ position: 'relative' }}>
      <SnippetCopyAction
        componentId="mlflow.eval-runs.start-run-modal.copy-snippet"
        copyText={codeSnippet}
        css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
      />
      <CodeSnippet theme={theme.isDarkMode ? 'duotoneDark' : 'light'} language="python">
        {codeSnippet}
      </CodeSnippet>
    </div>
  );

  return (
    <>
      <Button componentId="mlflow.eval-runs.start-run-button" icon={<ChartLineIcon />} onClick={() => setIsOpen(true)}>
        <FormattedMessage
          defaultMessage="Run evaluation"
          description="Label for a button that displays instructions for starting a new evaluation run"
        />
      </Button>
      <Modal
        componentId="mlflow.eval-runs.start-run-modal"
        title={
          <FormattedMessage defaultMessage="Run evaluation" description="Title for the run evaluation modal dialog" />
        }
        visible={isOpen}
        okText="Discard"
        footer={null}
        onCancel={() => setIsOpen(false)}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text>{evalInstructions}</Typography.Text>
          {evalCodeSnippet}
        </div>
      </Modal>
    </>
  );
};
