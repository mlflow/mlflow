import { ClockIcon, TokenIcon } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { TimelineTreeMetric } from '../ModelTraceExplorerPreferencesContext';
import { formatCompactNumber, getSpanTokenUsage } from '../ModelTraceTokenUsage.utils';
import { formatCostUSD, formatEstimatedCostUSD } from '../../CostUtils';
import { estimatedCostMessages } from '../../EstimatedCostValue';
import { spanTimeFormatter } from './TimelineTree.utils';
import { shouldEnableNewEvalRunsUi } from '../../../../../common/utils/FeatureUtils';

export interface TimelineTreeMetricValue {
  key: TimelineTreeMetric;
  value: string;
  title: string;
  icon?: React.ReactNode;
}

export interface TimelineTreeComparisonMetricValue {
  key: TimelineTreeMetric;
  label: string;
  baseline: string;
  candidate: string;
  change: string;
  unchanged: boolean;
  title: string;
  icon?: React.ReactNode;
}

export const getTimelineTreeComparisonMetricValues = (
  baselineNode: ModelTraceSpanNode | undefined,
  candidateNode: ModelTraceSpanNode | undefined,
  intl: IntlShape,
): TimelineTreeComparisonMetricValue[] => {
  const metrics: Array<{
    key: TimelineTreeMetric;
    label: string;
    baseline?: number;
    candidate?: number;
    format: (value: number) => string;
    icon?: React.ReactNode;
  }> = [
    {
      key: 'duration',
      label: intl.formatMessage({ defaultMessage: 'Duration', description: 'Trace comparison metric label' }),
      baseline: baselineNode ? baselineNode.end - baselineNode.start : undefined,
      candidate: candidateNode ? candidateNode.end - candidateNode.start : undefined,
      format: spanTimeFormatter,
      icon: <ClockIcon />,
    },
    {
      key: 'tokens',
      label: intl.formatMessage({ defaultMessage: 'Tokens', description: 'Trace comparison metric label' }),
      baseline: baselineNode ? getSpanTokenUsage(baselineNode)?.total_tokens : undefined,
      candidate: candidateNode ? getSpanTokenUsage(candidateNode)?.total_tokens : undefined,
      format: formatCompactNumber,
      icon: <TokenIcon />,
    },
    {
      key: 'cost',
      label: intl.formatMessage({ defaultMessage: 'Estimated cost', description: 'Trace comparison metric label' }),
      baseline: baselineNode?.cost?.total_cost,
      candidate: candidateNode?.cost?.total_cost,
      format: (value) => formatCostUSD(value, 4),
    },
  ];

  return metrics.flatMap((metric) => {
    if (metric.baseline === undefined && metric.candidate === undefined) {
      return [];
    }
    let change = '—';
    if (metric.baseline !== undefined && metric.candidate !== undefined) {
      const delta = metric.candidate - metric.baseline;
      const sign = delta > 0 ? '+' : delta < 0 ? '−' : '';
      const absoluteChange = `${sign}${metric.format(Math.abs(delta))}`;
      const percentage =
        metric.baseline === 0
          ? undefined
          : `${delta > 0 ? '+' : delta < 0 ? '−' : ''}${Math.abs((delta / metric.baseline) * 100).toFixed(1)}%`;
      change = percentage ? `${absoluteChange} (${percentage})` : absoluteChange;
    }
    const baseline = metric.baseline === undefined ? '—' : metric.format(metric.baseline);
    const candidate = metric.candidate === undefined ? '—' : metric.format(metric.candidate);
    const unchanged =
      metric.baseline !== undefined && metric.candidate !== undefined && metric.baseline === metric.candidate;
    return [
      {
        key: metric.key,
        label: metric.label,
        baseline,
        candidate,
        change,
        unchanged,
        title: unchanged
          ? intl.formatMessage(
              {
                defaultMessage: '{label}: {value}',
                description: 'Tooltip for an unchanged span metric in a trace comparison',
              },
              { label: metric.label, value: baseline },
            )
          : intl.formatMessage(
              {
                defaultMessage: '{label}: {baseline} → {candidate}; change {change}',
                description: 'Tooltip for a baseline and candidate span metric comparison',
              },
              { label: metric.label, baseline, candidate, change },
            ),
        icon: metric.icon,
      },
    ];
  });
};

export const getTimelineTreeMetricValue = (
  metric: TimelineTreeMetric,
  node: ModelTraceSpanNode,
  intl: IntlShape,
): TimelineTreeMetricValue | undefined => {
  const duration = spanTimeFormatter(node.end - node.start);
  switch (metric) {
    case 'duration':
      return {
        key: metric,
        value: duration,
        title: intl.formatMessage(
          { defaultMessage: 'Duration: {value}', description: 'Trace span duration metric tooltip' },
          { value: duration },
        ),
      };
    case 'tokens': {
      const tokens = getSpanTokenUsage(node)?.total_tokens;
      return tokens === undefined
        ? undefined
        : {
            key: metric,
            value: formatCompactNumber(tokens),
            title: intl.formatMessage(
              { defaultMessage: '{value} tokens', description: 'Trace span token metric tooltip' },
              { value: tokens },
            ),
            icon: <TokenIcon />,
          };
    }
    case 'cost': {
      if (!node.cost) {
        return undefined;
      }
      if (shouldEnableNewEvalRunsUi()) {
        // The new eval runs UI rounds span costs to 4 decimal places and marks them as estimated.
        const estimatedCost = formatEstimatedCostUSD(node.cost.total_cost, 4);
        return {
          key: metric,
          value: estimatedCost,
          // Combine the value and the explanation in the native title so hovering the compact
          // metric shows both, without nesting a DS Tooltip inside the parent span tooltip.
          title: intl.formatMessage(
            {
              defaultMessage: 'Estimated cost: {value}. {explanation}',
              description: 'Tooltip for span cost in the timeline tree, including the estimation explanation',
            },
            { value: estimatedCost, explanation: intl.formatMessage(estimatedCostMessages.tooltip) },
          ),
        };
      }
      const cost = formatCostUSD(node.cost.total_cost);
      return {
        key: metric,
        value: cost,
        title: intl.formatMessage(
          { defaultMessage: 'Estimated cost: {value}', description: 'Trace span estimated cost metric tooltip' },
          { value: cost },
        ),
      };
    }
  }
};
