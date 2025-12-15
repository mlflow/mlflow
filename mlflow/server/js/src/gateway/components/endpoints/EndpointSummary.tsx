import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { LongFormSummary } from '../../../common/components/long-form';
import { formatProviderName } from '../../utils/providerUtils';
import { formatMaskedValueSimple } from '../../utils/secretUtils';
import { formatTokens, formatCost } from '../../utils/formatters';
import type { Model, SecretInfo } from '../../types';

export interface EndpointSummaryProps {
  /** Provider name (e.g., 'openai', 'anthropic') */
  provider?: string;
  /** Model name (e.g., 'gpt-4', 'claude-3-opus') */
  modelName?: string;
  /** Model metadata for displaying capabilities and specs */
  modelMetadata?: Model;
  /** Secret/API key info for displaying connection details (used in 'configured' mode) */
  secret?: SecretInfo;
  /** Name of selected existing secret (used in 'existing' mode during create/edit) */
  selectedSecretName?: string;
  /** Whether to show the connection section (hide for create flow before secret is configured) */
  showConnection?: boolean;
  /** Connection display mode: 'configured' shows secret name, 'new' shows "New secret", 'existing' shows selected secret name */
  connectionMode?: 'configured' | 'new' | 'existing';
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
}

/**
 * Shared summary component for endpoint create, edit, and details pages.
 * Displays provider, model (with capabilities and connection/API key info).
 * Connection is nested under Model since each model can have its own API key mapping.
 */
export const EndpointSummary = ({
  provider,
  modelName,
  modelMetadata,
  secret,
  selectedSecretName,
  showConnection = true,
  connectionMode = 'configured',
  componentIdPrefix = 'mlflow.gateway.endpoint-summary',
}: EndpointSummaryProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Build capabilities list
  const capabilities: string[] = [];
  if (modelMetadata?.supports_function_calling) capabilities.push('Tools');
  if (modelMetadata?.supports_reasoning) capabilities.push('Reasoning');

  // Format model specs
  const contextWindow = formatTokens(modelMetadata?.max_input_tokens);
  const inputCost = formatCost(modelMetadata?.input_cost_per_token);
  const outputCost = formatCost(modelMetadata?.output_cost_per_token);

  // Render connection info
  const renderConnection = () => {
    if (connectionMode === 'configured' && secret) {
      return (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>{secret.secret_name}</Typography.Text>
          <Typography.Text
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              backgroundColor: theme.colors.tagDefault,
              padding: `2px ${theme.spacing.xs}px`,
              borderRadius: theme.general.borderRadiusBase,
              width: 'fit-content',
            }}
          >
            {formatMaskedValueSimple(secret.masked_value)}
          </Typography.Text>
        </div>
      );
    } else if (connectionMode === 'new') {
      return (
        <Typography.Text>
          <FormattedMessage defaultMessage="New API key" description="Summary new secret" />
        </Typography.Text>
      );
    } else if (connectionMode === 'existing') {
      if (selectedSecretName) {
        const truncatedName =
          selectedSecretName.length > 15 ? `${selectedSecretName.slice(0, 15)}…` : selectedSecretName;
        return (
          <Typography.Text css={{ paddingLeft: theme.spacing.sm }}>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="API Key:" description="Summary API key label" />
            </Typography.Text>{' '}
            {truncatedName}
          </Typography.Text>
        );
      }
      return (
        <Typography.Text css={{ paddingLeft: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="Existing API key" description="Summary existing secret" />
        </Typography.Text>
      );
    }
    return (
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
      </Typography.Text>
    );
  };

  return (
    <LongFormSummary
      title={intl.formatMessage({
        defaultMessage: 'Summary',
        description: 'Summary sidebar title',
      })}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Provider */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold color="secondary">
            <FormattedMessage defaultMessage="Provider" description="Summary provider label" />
          </Typography.Text>
          {provider ? (
            <Tag componentId={`${componentIdPrefix}.provider`}>{formatProviderName(provider)}</Tag>
          ) : (
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
            </Typography.Text>
          )}
        </div>

        {/* Model (with nested Connection) */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold color="secondary">
            <FormattedMessage defaultMessage="Model" description="Summary model label" />
          </Typography.Text>
          {modelName ? (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                paddingLeft: theme.spacing.sm,
                borderLeft: `2px solid ${theme.colors.borderDecorative}`,
              }}
            >
              {/* Model name - styled div for proper text wrapping */}
              <div
                css={{
                  backgroundColor: theme.colors.tagDefault,
                  padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
                  borderRadius: theme.borders.borderRadiusMd,
                  fontSize: theme.typography.fontSizeSm,
                  wordBreak: 'break-all',
                  overflowWrap: 'anywhere',
                  width: 'fit-content',
                }}
              >
                {modelName}
              </div>

              {/* Capabilities */}
              {capabilities.length > 0 && (
                <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
                  {capabilities.map((cap) => (
                    <Tag key={cap} color="turquoise" componentId={`${componentIdPrefix}.capability.${cap}`}>
                      {cap}
                    </Tag>
                  ))}
                </div>
              )}

              {/* Context & Cost info */}
              {modelMetadata && (contextWindow !== '-' || inputCost !== '-' || outputCost !== '-') && (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                    fontSize: theme.typography.fontSizeSm,
                    color: theme.colors.textSecondary,
                  }}
                >
                  {contextWindow !== '-' && (
                    <span>
                      {intl.formatMessage(
                        { defaultMessage: 'Context: {tokens}', description: 'Context window size' },
                        { tokens: contextWindow },
                      )}
                    </span>
                  )}
                  {(inputCost !== '-' || outputCost !== '-') && (
                    <span>
                      {intl.formatMessage(
                        { defaultMessage: 'Cost: {input} in / {output} out', description: 'Model cost per token' },
                        { input: inputCost, output: outputCost },
                      )}
                    </span>
                  )}
                </div>
              )}

              {/* Connection (nested under model) */}
              {showConnection && (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  <Typography.Text bold color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                    <FormattedMessage defaultMessage="Connection" description="Summary connection/API key label" />
                  </Typography.Text>
                  {renderConnection()}
                </div>
              )}
            </div>
          ) : (
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Not configured" description="Summary not configured" />
            </Typography.Text>
          )}
        </div>
      </div>
    </LongFormSummary>
  );
};
