import { useMemo } from 'react';
import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { formatProviderName, formatAuthMethodName, formatCredentialFieldName } from '../../utils/providerUtils';
import { parseAuthConfig } from '../../utils/secretUtils';
import { formatTokens, formatCost } from '../../utils/formatters';
import { useSecretQuery } from '../../hooks/useSecretQuery';
import { MaskedValueDisplay } from '../secrets/MaskedValueDisplay';
import type { ModelDefinition, ProviderModel, SecretInfo } from '../../types';

const AuthConfigDisplay = ({ secret }: { secret: SecretInfo | undefined }) => {
  const { theme } = useDesignSystemTheme();

  const authConfig = useMemo(() => parseAuthConfig(secret), [secret]);

  if (!authConfig) return null;
  const configEntries = Object.entries(authConfig).filter(([key]) => key !== 'auth_mode');
  if (configEntries.length === 0) return null;

  return (
    <>
      <Typography.Text color="secondary">
        <FormattedMessage defaultMessage="Config:" description="Auth config label" />
      </Typography.Text>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        {configEntries.map(([key, value]) => (
          <div key={key}>
            <Typography.Text color="secondary">{formatCredentialFieldName(key)}: </Typography.Text>
            <Typography.Text css={{ fontFamily: 'monospace' }}>{String(value)}</Typography.Text>
          </div>
        ))}
      </div>
    </>
  );
};

interface ModelCardProps {
  modelDefinition: ModelDefinition | undefined;
  modelMetadata: ProviderModel | undefined;
  onKeyClick?: (secret: SecretInfo) => void;
}

export const ModelCard = ({ modelDefinition, modelMetadata, onKeyClick }: ModelCardProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const hasSecretId = Boolean(modelDefinition?.secret_id);

  const { data: secretData, isLoading: isSecretLoading } = useSecretQuery(
    hasSecretId ? modelDefinition?.secret_id : undefined,
  );

  const contextWindow = useMemo(
    () => formatTokens(modelMetadata?.max_input_tokens ?? null),
    [modelMetadata?.max_input_tokens],
  );
  const inputCost = useMemo(
    () => formatCost(modelMetadata?.input_cost_per_token ?? null),
    [modelMetadata?.input_cost_per_token],
  );
  const outputCost = useMemo(
    () => formatCost(modelMetadata?.output_cost_per_token ?? null),
    [modelMetadata?.output_cost_per_token],
  );

  const authConfig = useMemo(() => parseAuthConfig(secretData?.secret), [secretData?.secret]);

  if (!modelDefinition) {
    return null;
  }

  const secret = secretData?.secret;

  return (
    <div
      css={{
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'auto 1fr',
          gap: `${theme.spacing.xs}px ${theme.spacing.md}px`,
          alignItems: 'baseline',
        }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="Model:" description="Model name label" />
        </Typography.Text>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexWrap: 'wrap' }}>
          <Typography.Text css={{ fontFamily: 'monospace' }}>
            {modelDefinition.provider ? `${formatProviderName(modelDefinition.provider)} / ` : ''}
            {modelDefinition.model_name}
          </Typography.Text>
        </div>

        {modelMetadata && (contextWindow !== '-' || inputCost !== '-' || outputCost !== '-') && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Specs:" description="Model specs label" />
            </Typography.Text>
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              {[
                contextWindow !== '-' &&
                  intl.formatMessage(
                    { defaultMessage: 'Context: {tokens}', description: 'Context window size' },
                    { tokens: contextWindow },
                  ),
                inputCost !== '-' &&
                  intl.formatMessage(
                    { defaultMessage: 'Input: {cost}', description: 'Input cost' },
                    { cost: inputCost },
                  ),
                outputCost !== '-' &&
                  intl.formatMessage(
                    { defaultMessage: 'Output: {cost}', description: 'Output cost' },
                    { cost: outputCost },
                  ),
              ]
                .filter(Boolean)
                .join(' • ')}
            </Typography.Text>
          </>
        )}

        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="API Key:" description="API key name label" />
        </Typography.Text>
        {!hasSecretId ? (
          <Tag color="coral" componentId="mlflow.gateway.endpoint-details.no-api-key">
            <FormattedMessage defaultMessage="No API key configured" description="No API key configured message" />
          </Tag>
        ) : isSecretLoading ? (
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Loading..." description="Loading secret" />
          </Typography.Text>
        ) : secret ? (
          <span
            role="button"
            tabIndex={0}
            onClick={() => onKeyClick?.(secret)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                onKeyClick?.(secret);
              }
            }}
            css={{
              color: theme.colors.actionPrimaryBackgroundDefault,
              fontWeight: theme.typography.typographyBoldFontWeight,
              cursor: 'pointer',
              '&:hover': {
                textDecoration: 'underline',
              },
            }}
          >
            {secret.secret_name}
          </span>
        ) : (
          <Tag color="coral" componentId="mlflow.gateway.endpoint-details.api-key-not-found">
            <FormattedMessage defaultMessage="API key not found" description="API key not found message" />
          </Tag>
        )}

        {authConfig?.['auth_mode'] && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Auth Type:" description="Auth type label" />
            </Typography.Text>
            <Typography.Text>{formatAuthMethodName(String(authConfig['auth_mode']))}</Typography.Text>
          </>
        )}

        {secret?.masked_values && Object.keys(secret.masked_values).length > 0 && (
          <>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Masked Key:" description="Masked API key label" />
            </Typography.Text>
            <MaskedValueDisplay maskedValue={secret.masked_values} compact />
          </>
        )}

        <AuthConfigDisplay secret={secret} />
      </div>
    </div>
  );
};
