import { useMemo } from 'react';

import { HoverCard, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { ModelTraceSpanNode, SpanCostInfo } from '../ModelTrace.types';
import { formatCostUSD, formatEstimatedCostUSD } from '../CostUtils';
import { estimatedCostMessages } from '../EstimatedCostValue';
import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';

const SpanCostHoverCard = ({ cost }: { cost: SpanCostInfo }) => {
  const { theme } = useDesignSystemTheme();

  const intl = useIntl();
  // The new eval runs UI marks pricing-table costs as estimated (≈ prefix and an explanation).
  const isNewEvalRunsUi = shouldEnableNewEvalRunsUi();
  const totalCost = useMemo(
    () => (isNewEvalRunsUi ? formatEstimatedCostUSD(cost.total_cost) : formatCostUSD(cost.total_cost)),
    [cost.total_cost, isNewEvalRunsUi],
  );
  const inputCost = useMemo(() => formatCostUSD(cost.input_cost), [cost.input_cost]);
  const outputCost = useMemo(() => formatCostUSD(cost.output_cost), [cost.output_cost]);
  // aria-label announces the plain dollar amount so screen readers say "Estimated cost: $X" rather than the ≈ glyph.
  const estimatedAriaLabel = isNewEvalRunsUi
    ? intl.formatMessage(estimatedCostMessages.ariaLabel, { cost: formatCostUSD(cost.total_cost) })
    : undefined;

  return (
    <HoverCard
      trigger={
        <div
          aria-label={estimatedAriaLabel}
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'row',
            gap: theme.spacing.sm,
          }}
        >
          <Typography.Text size="md" color="secondary">
            <FormattedMessage defaultMessage="Cost" description="Label for cost in span details" />
          </Typography.Text>
          <Tag componentId="shared.model-trace-explorer.span-cost-badge" color="lime">
            <span>{totalCost}</span>
          </Tag>
        </div>
      }
      content={
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
            padding: theme.spacing.sm,
            maxWidth: 400,
          }}
        >
          <Typography.Title level={3} withoutMargins>
            <FormattedMessage defaultMessage="Cost breakdown" description="Header for span cost breakdown" />
          </Typography.Title>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
            }}
          >
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <Typography.Text size="md">
                <FormattedMessage defaultMessage="Input cost" description="Label for input cost" />
              </Typography.Text>
              <Tag componentId="shared.model-trace-explorer.span-cost-hovercard.input-cost.tag">
                <span>{inputCost}</span>
              </Tag>
            </div>
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <Typography.Text size="md">
                <FormattedMessage defaultMessage="Output cost" description="Label for output cost" />
              </Typography.Text>
              <Tag componentId="shared.model-trace-explorer.span-cost-hovercard.output-cost.tag">
                <span>{outputCost}</span>
              </Tag>
            </div>
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
                paddingTop: theme.spacing.sm,
                borderTop: `1px solid ${theme.colors.borderDecorative}`,
              }}
            >
              <Typography.Text size="md" bold>
                <FormattedMessage defaultMessage="Total" description="Label for total cost" />
              </Typography.Text>
              <Tag componentId="shared.model-trace-explorer.span-cost-hovercard.total-cost.tag">
                <span>{totalCost}</span>
              </Tag>
            </div>
          </div>
          {isNewEvalRunsUi && (
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage {...estimatedCostMessages.tooltip} />
            </Typography.Text>
          )}
        </div>
      }
      side="bottom"
      align="start"
    />
  );
};

export const SpanModelCostBadge = ({
  activeSpan,
  className,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
}) => {
  const { theme } = useDesignSystemTheme();

  if (!activeSpan) {
    return null;
  }

  const { modelName, cost } = activeSpan;

  // Don't render anything if there's no model or cost
  if (!modelName && !cost) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: theme.spacing.sm,
        paddingLeft: theme.spacing.xs,
        flexWrap: 'wrap',
      }}
      className={className}
    >
      {modelName && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text size="md" color="secondary">
            <FormattedMessage defaultMessage="Model" description="Label for model name in span details" />
          </Typography.Text>
          <Tag componentId="shared.model-trace-explorer.span-model-badge" color="turquoise">
            {modelName}
          </Tag>
        </div>
      )}
      {cost && <SpanCostHoverCard cost={cost} />}
    </div>
  );
};
