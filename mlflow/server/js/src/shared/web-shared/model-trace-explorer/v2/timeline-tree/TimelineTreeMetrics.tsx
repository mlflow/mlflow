import { TokenIcon } from '@databricks/design-system';
import type { IntlShape } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { TimelineTreeMetric } from '../ModelTraceExplorerPreferencesContext';
import { formatCompactNumber, getSpanTokenUsage } from '../ModelTraceTokenUsage.utils';
import { formatCostUSD } from '../../CostUtils';
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
      // The new eval runs UI rounds span costs to 4 decimal places.
      const cost = shouldEnableNewEvalRunsUi()
        ? formatCostUSD(node.cost.total_cost, 4)
        : formatCostUSD(node.cost.total_cost);
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
