import { useState } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { AggregationType, MetricViewType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage } from 'react-intl';

import { AgentActionCard } from '../../components/onboarding/AgentActionCard';
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

  // Only fire the trace-count query once the user actually opens the Python tab — most
  // empty-state visitors click "Evaluate traces" or the coding-agent path and never need it.
  const [hasOpenedPythonTab, setHasOpenedPythonTab] = useState(false);

  const { data: traceMetrics, isLoading: isTraceMetricsLoading } = useTraceMetricsQuery({
    experimentIds: [experimentId],
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    enabled: hasOpenedPythonTab,
  });
  const traceCount = Number(traceMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT] ?? 0);
  const hasTraces = traceCount > 0;

  // The Python tab is always shown so it doesn't pop in and steal clicks. Default to the
  // dataset variant until the query confirms there are traces — if a user copies the snippet
  // in the first few ms before the count returns, the dataset path still produces a runnable
  // eval. With traces present we swap to the trace-based snippet.
  const codeSnippet = {
    content: hasTraces ? getTraceCodeSnippet(experimentId) : getDatasetCodeSnippet(experimentId),
    language: 'python' as const,
    label: PYTHON_TAB_LABEL,
    // Until the trace-count query resolves we don't know which snippet variant to show, so render a
    // skeleton rather than the dataset default that could flip to the trace variant under a fast copier.
    isLoading: hasOpenedPythonTab && isTraceMetricsLoading,
  };

  const header = (
    <>
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <RunEvaluationButton experimentId={experimentId} />
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
        onActiveTabChange={(tab) => {
          if (tab === 'code-snippet') {
            setHasOpenedPythonTab(true);
          }
        }}
      />
    </div>
  );
};
