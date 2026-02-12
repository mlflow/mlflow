import { useMemo } from 'react';

import { HoverCard, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SpanCostInfo } from '../ModelTrace.types';
import { formatCostUSD } from '../CostUtils';

const SpanCostHoverCard = ({ cost }: { cost: SpanCostInfo }) => {
  const { theme } = useDesignSystemTheme();

  const totalCost = useMemo(() => formatCostUSD(cost.total_cost), [cost.total_cost]);
  const inputCost = useMemo(() => formatCostUSD(cost.input_cost), [cost.input_cost]);
  const outputCost = useMemo(() => formatCostUSD(cost.output_cost), [cost.output_cost]);

  return (
    <HoverCard
      trigger={
        <div
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
        </div>
      }
      side="bottom"
      align="start"
    />
  );
};

export const SpanModelCostBadge = ({ activeSpan }: { activeSpan: ModelTraceSpanNode }) => {
  const { theme } = useDesignSystemTheme();

  const { modelName, cost } = activeSpan;

  // Don't render anything if there's no model or cost info
  if (!modelName && !cost) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        gap: theme.spacing.md,
        paddingLeft: theme.spacing.md,
        paddingBottom: theme.spacing.sm,
        flexWrap: 'wrap',
      }}
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
