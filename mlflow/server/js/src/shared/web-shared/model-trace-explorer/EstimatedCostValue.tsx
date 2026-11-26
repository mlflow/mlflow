import { Tooltip } from '@databricks/design-system';
import { defineMessages, FormattedMessage, useIntl, type MessageDescriptor } from '@databricks/i18n';

import { formatCostUSD, formatEstimatedCostUSD } from './CostUtils';

const COMPONENT_ID = 'mlflow.model_trace_explorer.estimated_cost_value';

export const estimatedCostMessages: Record<'tooltip' | 'ariaLabel', MessageDescriptor> = defineMessages({
  tooltip: {
    defaultMessage: 'Cost is estimated from token usage and public model pricing.',
    description: 'Tooltip explaining that cost is estimated, not reported by the provider',
  },
  ariaLabel: {
    defaultMessage: 'Estimated cost: {cost}',
    description:
      'Accessible name for the ≈-prefixed cost value; {cost} is the plain dollar amount (no ≈ glyph) so screen readers announce it as "Estimated cost: $X" rather than reading the glyph literally',
  },
});

export interface EstimatedCostValueProps {
  cost: number;
  maximumFractionDigits?: number;
  className?: string;
}

/**
 * Renders a ≈-prefixed cost value with a tooltip explaining the estimation source.
 * The aria-label announces "Estimated cost: <amount>" so screen readers do not read the ≈ glyph literally.
 */
export const EstimatedCostValue = ({
  cost,
  maximumFractionDigits,
  className,
}: EstimatedCostValueProps): JSX.Element => {
  const intl = useIntl();
  const formatted = formatEstimatedCostUSD(cost, maximumFractionDigits);
  const ariaLabel = intl.formatMessage(estimatedCostMessages.ariaLabel, {
    cost: formatCostUSD(cost, maximumFractionDigits),
  });

  return (
    <Tooltip componentId={`${COMPONENT_ID}.tooltip`} content={<FormattedMessage {...estimatedCostMessages.tooltip} />}>
      <span aria-label={ariaLabel} className={className}>
        {formatted}
      </span>
    </Tooltip>
  );
};
