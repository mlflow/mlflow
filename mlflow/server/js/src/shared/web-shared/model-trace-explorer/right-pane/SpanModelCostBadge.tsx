import { useMemo } from 'react';

import { HoverCard, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SpanCostInfo } from '../ModelTrace.types';
import { formatCostUSD } from '../CostUtils';

const SpanCostHoverCard = ({ cost }: { cost: SpanCostInfo }) => {
  const { theme } = useDesignSystemTheme();

  const totalCost = useMemo(() => formatCostUSD(cost.total_cost), [cost.total_cost]);

  // Build breakdown items for all non-zero cost components
  const breakdownItems = useMemo(() => {
    const items: Array<{ key: string; label: string; value: string }> = [];

    if ((cost.input_cost ?? 0) > 0) {
      items.push({
        key: 'input',
        label: 'Input cost',
        value: formatCostUSD(cost.input_cost!),
      });
    }

    if ((cost.output_cost ?? 0) > 0) {
      items.push({
        key: 'output',
        label: 'Output cost',
        value: formatCostUSD(cost.output_cost!),
      });
    }

    if ((cost.tool_cost ?? 0) > 0) {
      items.push({
        key: 'tool',
        label: 'Tool cost',
        value: formatCostUSD(cost.tool_cost!),
      });
    }

    if ((cost.embedding_cost ?? 0) > 0) {
      items.push({
        key: 'embedding',
        label: 'Embedding cost',
        value: formatCostUSD(cost.embedding_cost!),
      });
    }

    if ((cost.retrieval_cost ?? 0) > 0) {
      items.push({
        key: 'retrieval',
        label: 'Retrieval cost',
        value: formatCostUSD(cost.retrieval_cost!),
      });
    }

    if ((cost.other_cost ?? 0) > 0) {
      items.push({
        key: 'misc',
        label: 'Other cost',
        value: formatCostUSD(cost.other_cost!),
      });
    }

    return items;
  }, [cost]);

  const hasBreakdown = breakdownItems.length > 0;

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
            {hasBreakdown ? (
              <FormattedMessage defaultMessage="Cost breakdown" description="Header for span cost breakdown" />
            ) : (
              <FormattedMessage defaultMessage="Cost" description="Header for span cost" />
            )}
          </Typography.Title>
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
            }}
          >
            {breakdownItems.map((item) => (
              <div
                key={item.key}
                css={{
                  display: 'flex',
                  flexDirection: 'row',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Typography.Text size="md">{item.label}</Typography.Text>
                <Tag componentId={`shared.model-trace-explorer.span-cost-hovercard.${item.key}-cost.tag`}>
                  <span>{item.value}</span>
                </Tag>
              </div>
            ))}
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
                paddingTop: hasBreakdown ? theme.spacing.sm : 0,
                borderTop: hasBreakdown ? `1px solid ${theme.colors.borderDecorative}` : 'none',
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
