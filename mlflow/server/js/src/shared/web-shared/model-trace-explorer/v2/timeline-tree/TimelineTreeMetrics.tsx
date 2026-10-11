import { TokenIcon } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { TimelineTreeMetric } from '../ModelTraceExplorerPreferencesContext';
import { getSpanTokenUsage } from '../ModelTraceTokenUsage.utils';
import { formatCostUSD } from '../../CostUtils';
import { spanTimeFormatter } from './TimelineTree.utils';

export interface TimelineTreeMetricValue {
  key: TimelineTreeMetric;
  value: string;
  title: string;
  icon?: React.ReactNode;
}

const formatCompactNumber = (value: number) =>
  new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);

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
    case 'cost':
      return node.cost
        ? {
            key: metric,
            value: formatCostUSD(node.cost.total_cost),
            title: intl.formatMessage(
              { defaultMessage: 'Estimated cost: {value}', description: 'Trace span estimated cost metric tooltip' },
              { value: formatCostUSD(node.cost.total_cost) },
            ),
          }
        : undefined;
  }
};
