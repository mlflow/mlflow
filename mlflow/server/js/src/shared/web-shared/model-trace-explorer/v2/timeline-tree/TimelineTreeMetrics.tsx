import { TokenIcon } from '@databricks/design-system';
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
