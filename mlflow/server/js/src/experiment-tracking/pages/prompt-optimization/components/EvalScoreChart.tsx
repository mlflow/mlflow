import { useMemo } from 'react';
import { Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchEndpoint, jsonBigIntResponseParser } from '../../../../common/utils/FetchUtils';
import { LazyPlot } from '../../../components/LazyPlot';

interface EvalScoreChartProps {
  runId: string;
  scorerNames?: string[];
}

interface MetricHistoryEntry {
  key: string;
  value: number;
  timestamp: number;
  step: number;
  run_id: string;
}

interface MetricHistoryBulkResponse {
  metrics: MetricHistoryEntry[];
}

/**
 * Component to display eval_score metric history chart for GEPA optimization jobs.
 * Shows eval_score (aggregate) and eval_score.<scorer_name> metrics over optimization steps.
 */
export const EvalScoreChart = ({ runId, scorerNames }: EvalScoreChartProps) => {
  const { theme } = useDesignSystemTheme();

  // Build metric keys to fetch - use lowercase scorer names as that's what GEPA logs
  const metricKeys = useMemo(() => {
    const keys = ['eval_score'];
    if (scorerNames) {
      scorerNames.forEach((scorer) => {
        // GEPA optimizer logs metrics with lowercase scorer names
        keys.push(`eval_score.${scorer.toLowerCase()}`);
      });
    }
    return keys;
  }, [scorerNames]);

  // Fetch metric history for all eval_score metrics using the bulk interval endpoint
  const { data: metricsData, isLoading } = useQuery<Record<string, MetricHistoryEntry[]>, Error>(
    ['optimization_eval_scores_history', runId, metricKeys],
    {
      queryFn: async () => {
        const results: Record<string, MetricHistoryEntry[]> = {};

        // Fetch each metric using the bulk-interval endpoint (same as run charts use)
        await Promise.all(
          metricKeys.map(async (metricKey) => {
            try {
              const queryParams = new URLSearchParams();
              queryParams.append('run_ids', runId);
              queryParams.append('metric_key', metricKey);

              const response = (await fetchEndpoint({
                relativeUrl: `ajax-api/2.0/mlflow/metrics/get-history-bulk-interval?${queryParams.toString()}`,
                success: jsonBigIntResponseParser,
              })) as MetricHistoryBulkResponse;

              if (response?.metrics && response.metrics.length > 0) {
                // Sort by step
                results[metricKey] = response.metrics.sort((a, b) => a.step - b.step);
              }
            } catch {
              // Metric might not exist - this is expected if the optimizer hasn't logged it yet
            }
          }),
        );

        return results;
      },
      enabled: !!runId,
      retry: false,
    },
  );

  // Get the actual metric keys that have data
  const evalScoreMetricKeys = useMemo(() => {
    if (!metricsData) return [];
    return Object.keys(metricsData);
  }, [metricsData]);

  // Define colors for different metrics
  const metricColors: Record<string, string> = useMemo(() => {
    const colors: Record<string, string> = {
      eval_score: '#6B47DC', // Purple for aggregate
    };

    // Use distinct colors for each scorer
    const scorerColors = [
      theme.colors.blue500,
      theme.colors.green500,
      theme.colors.yellow500,
      '#E91E63', // Pink
      '#00BCD4', // Turquoise
    ];

    // Assign colors to per-scorer metrics
    evalScoreMetricKeys
      .filter((key) => key !== 'eval_score')
      .forEach((key, index) => {
        colors[key] = scorerColors[index % scorerColors.length];
      });

    return colors;
  }, [evalScoreMetricKeys, theme]);

  // Prepare plot data
  const plotData = useMemo(() => {
    if (!metricsData) return [];

    return Object.entries(metricsData).map(([metricKey, entries]) => {
      const displayName = metricKey === 'eval_score' ? 'Aggregate Score' : metricKey.replace('eval_score.', '');

      return {
        x: entries.map((e) => e.step),
        y: entries.map((e) => e.value),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        name: displayName,
        line: {
          color: metricColors[metricKey] || theme.colors.textPrimary,
          width: metricKey === 'eval_score' ? 3 : 2,
        },
        marker: {
          size: metricKey === 'eval_score' ? 8 : 6,
        },
      };
    });
  }, [metricsData, metricColors, theme]);

  const layout = useMemo(
    () => ({
      title: undefined,
      xaxis: {
        title: 'Optimization Step',
        tickfont: { size: 11, color: theme.colors.textSecondary },
        gridcolor: theme.colors.borderDecorative,
      },
      yaxis: {
        title: 'Score',
        tickfont: { size: 11, color: theme.colors.textSecondary },
        gridcolor: theme.colors.borderDecorative,
        range: [0, 1.05], // Scores are typically 0-1
      },
      margin: { l: 50, r: 20, t: 20, b: 50 },
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        y: 1.02,
        x: 0,
        xanchor: 'left' as const,
        yanchor: 'bottom' as const,
      },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        color: theme.colors.textPrimary,
      },
    }),
    [theme],
  );

  const config = useMemo(
    () => ({
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'] as const,
    }),
    [],
  );

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, padding: theme.spacing.md }}>
        <Spinner size="small" />
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="Loading score history..." description="Loading scores message" />
        </Typography.Text>
      </div>
    );
  }

  if (!metricsData || Object.keys(metricsData).length === 0) {
    return (
      <div>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No evaluation score history available. Scores will appear once the optimizer begins validating candidates."
            description="No scores available message"
          />
        </Typography.Text>
        <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginTop: theme.spacing.xs }}>
          (Looking for metrics: {metricKeys.join(', ')})
        </Typography.Text>
      </div>
    );
  }

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.md,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
    >
      <div css={{ height: 300 }}>
        <LazyPlot data={plotData} layout={layout} config={config} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
};
