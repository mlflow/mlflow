import { isObject } from 'lodash';
import { useMemo } from 'react';

import { HoverCard, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { formatCostUSD } from './CostUtils';

export interface TraceCost {
  input_cost: number;
  output_cost: number;
  total_cost: number;
}

export const isTraceCostType = (value?: unknown): value is TraceCost => {
  return (
    value !== undefined && isObject(value) && 'input_cost' in value && 'output_cost' in value && 'total_cost' in value
  );
};

export const ModelTraceExplorerCostHoverCard = ({ cost }: { cost: TraceCost }) => {
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
            <FormattedMessage defaultMessage="Cost" description="Label for the cost section" />
          </Typography.Text>
          <Tag componentId="shared.model-trace-explorer.header-details.cost.tag" color="lime">
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
            <FormattedMessage defaultMessage="Cost breakdown" description="Header for cost breakdown" />
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
              <Tag componentId="shared.model-trace-explorer.cost-hovercard.input-cost.tag">
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
              <Tag componentId="shared.model-trace-explorer.cost-hovercard.output-cost.tag">
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
              <Tag componentId="shared.model-trace-explorer.cost-hovercard.total-cost.tag">
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
