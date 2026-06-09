import { useDesignSystemTheme } from '@databricks/design-system';
import { AggregationType, MetricViewType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage } from 'react-intl';

import { AgentActionCard, type AgentActionCardCodeSnippet } from '../../components/onboarding/AgentActionCard';
import { useTraceMetricsQuery } from '../experiment-overview/hooks/useTraceMetricsQuery';

import { buildEvaluateAssistantPrompt, buildEvaluatePrompt } from './evalRunsAgentPrompt';
import { getDatasetCodeSnippet, getTraceCodeSnippet } from './evaluationCodeSnippets';
import { RunEvaluationButton } from './RunEvaluationButton';

const PYTHON_TAB_LABEL = (
  <FormattedMessage
    defaultMessage="Python"
    description="Tab label for the runnable Python code-snippet path in the eval-runs empty state"
  />
);

export const EvalRunsEmptyStateCard = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();

  // Pick between the trace-based and dataset-based snippet based on whether this experiment
  // has any traces yet. Until the query resolves we don't render the Python tab to avoid
  // briefly showing the wrong variant.
  const { data: traceMetrics, isSuccess: isTraceMetricsLoaded } = useTraceMetricsQuery({
    experimentIds: [experimentId],
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
  });
  const traceCount = Number(traceMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT] ?? 0);
  const hasTraces = traceCount > 0;

  const codeSnippet: AgentActionCardCodeSnippet | undefined = isTraceMetricsLoaded
    ? {
        content: hasTraces ? getTraceCodeSnippet(experimentId) : getDatasetCodeSnippet(experimentId),
        language: 'python',
        label: PYTHON_TAB_LABEL,
      }
    : undefined;

  const header = (
    <>
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <RunEvaluationButton
          experimentId={experimentId}
          type="primary"
          label={
            <FormattedMessage
              defaultMessage="Evaluate traces"
              description="Primary CTA in the eval-runs empty state — opens the trace selection + scorer picker modal"
            />
          }
        />
      </div>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.md,
          color: theme.colors.textSecondary,
          fontSize: 12,
        }}
      >
        <div css={{ flex: 1, height: 1, backgroundColor: theme.colors.border }} />
        <FormattedMessage
          defaultMessage="Or use coding agents/code snippets"
          description="Divider label separating the primary UI evaluation flow from the code-based alternatives"
        />
        <div css={{ flex: 1, height: 1, backgroundColor: theme.colors.border }} />
      </div>
    </>
  );

  return (
    <div css={{ width: '100%', maxWidth: 720, margin: '0 auto' }}>
      <AgentActionCard
        header={header}
        codingAgentPrompt={buildEvaluatePrompt(experimentId)}
        assistantPrompt={buildEvaluateAssistantPrompt(experimentId)}
        componentId="mlflow.eval-runs.empty-state.unified"
        codeSnippet={codeSnippet}
      />
    </div>
  );
};
