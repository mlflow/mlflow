import { isObject } from 'lodash';
import { useMemo } from 'react';

import { HoverCard, Tag, TokenIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { truncateToFirstLineWithMaxLength } from './TagUtils';

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export const isTokenUsageType = (value?: unknown): value is TokenUsage => {
  return (
    value !== undefined &&
    isObject(value) &&
    'input_tokens' in value &&
    'output_tokens' in value &&
    'total_tokens' in value
  );
};

export const ModelTraceExplorerTokenUsageHoverCard = ({ tokenUsage }: { tokenUsage: TokenUsage }) => {
  const { theme } = useDesignSystemTheme();

  const totalTokens = useMemo(() => tokenUsage.total_tokens.toString(), [tokenUsage]);
  const inputTokens = useMemo(() => tokenUsage.input_tokens.toString(), [tokenUsage]);
  const outputTokens = useMemo(() => tokenUsage.output_tokens.toString(), [tokenUsage]);

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
            <FormattedMessage defaultMessage="Token count" description="Label for the token count section" />
          </Typography.Text>
          <Tag componentId="shared.model-trace-explorer.header-details.tag">
            <span css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: theme.spacing.xs }}>
              <span>
                <TokenIcon />
              </span>
              <span>{truncateToFirstLineWithMaxLength(totalTokens, 40)}</span>
            </span>
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
            <FormattedMessage defaultMessage="Usage breakdown" description="Header for token usage breakdown" />
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
                <FormattedMessage defaultMessage="Input tokens" description="Label for input token usage" />
              </Typography.Text>
              <Tag componentId="shared.model-trace-explorer.token-usage-hovercard.input-tokens.tag">
                <span>{inputTokens}</span>
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
                <FormattedMessage defaultMessage="Output tokens" description="Label for output token usage" />
              </Typography.Text>
              <Tag componentId="shared.model-trace-explorer.token-usage-hovercard.output-tokens.tag">
                <span>{outputTokens}</span>
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
                <FormattedMessage defaultMessage="Total" description="Label for total token usage" />
              </Typography.Text>
              <Tag componentId="shared.model-trace-explorer.token-usage-hovercard.total-tokens.tag">
                <span>{totalTokens}</span>
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
