import { useMemo } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { formatProviderName, formatAuthMethodName, formatSecretFieldName } from '../../utils/providerUtils';
import { parseAuthConfig, parseMaskedValues, isSingleMaskedValue } from '../../utils/secretUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import type { SecretInfo } from '../../types';

interface SecretDetailsProps {
  secret: SecretInfo;
  /** Whether to show as a card with background (default: true) */
  showCard?: boolean;
}

/**
 * Displays detailed information about a secret/API key.
 * Can be used standalone or within a drawer/modal.
 */
export const SecretDetails = ({ secret, showCard = true }: SecretDetailsProps) => {
  const { theme } = useDesignSystemTheme();

  // Parse auth config and masked values using shared utils
  const authConfig = useMemo(() => parseAuthConfig(secret), [secret]);
  const maskedValues = useMemo(() => parseMaskedValues(secret), [secret]);

  const content = (
    <div css={{ display: 'flex', flexDirection: 'column', paddingTop: theme.spacing.md }}>
      {/* Secret name */}
      <Typography.Title level={3} css={{ margin: 0 }}>
        {secret.secret_name}
      </Typography.Title>

      {/* Metadata section */}
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginTop: theme.spacing.lg }}>
        {/* Provider */}
        {secret.provider && (
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Provider:" description="Secret provider label" />
            </Typography.Text>
            <Typography.Text>{formatProviderName(secret.provider)}</Typography.Text>
          </div>
        )}

        {/* Auth Type - only show if auth_mode is set in auth_config (indicates multi-auth provider) */}
        {authConfig?.['auth_mode'] && (
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Auth Type:" description="Auth type label" />
            </Typography.Text>
            <Typography.Text>{formatAuthMethodName(String(authConfig['auth_mode']))}</Typography.Text>
          </div>
        )}

        {/* Auth Config - show non-encrypted configuration if present (excluding auth_mode which is shown above) */}
        {authConfig && Object.keys(authConfig).filter((key) => key !== 'auth_mode').length > 0 && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text color="secondary">
              <FormattedMessage defaultMessage="Config:" description="Auth config label" />
            </Typography.Text>
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                paddingLeft: theme.spacing.md,
              }}
            >
              {Object.entries(authConfig)
                .filter(([key]) => key !== 'auth_mode')
                .map(([key, value]) => (
                  <div key={key} css={{ display: 'flex', gap: theme.spacing.xs }}>
                    <Typography.Text color="secondary">{formatSecretFieldName(key)}:</Typography.Text>
                    <Typography.Text css={{ fontFamily: 'monospace' }}>{String(value)}</Typography.Text>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Masked values */}
        {maskedValues && maskedValues.length > 0 && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text color="secondary">
              {isSingleMaskedValue(maskedValues) ? (
                <FormattedMessage defaultMessage="Masked Key:" description="Masked API key label (singular)" />
              ) : (
                <FormattedMessage defaultMessage="Masked Keys:" description="Masked API keys section label" />
              )}
            </Typography.Text>
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                paddingLeft: theme.spacing.md,
              }}
            >
              {maskedValues.map(([key, value], index) =>
                key === '' ? (
                  // Single value without key label
                  <Typography.Text
                    key={index}
                    css={{
                      fontFamily: 'monospace',
                      backgroundColor: theme.colors.tagDefault,
                      padding: `2px ${theme.spacing.xs}px`,
                      borderRadius: theme.general.borderRadiusBase,
                      width: 'fit-content',
                    }}
                  >
                    {value}
                  </Typography.Text>
                ) : (
                  // Multiple values with key labels
                  <div key={key} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <Typography.Text color="secondary">{formatSecretFieldName(key)}:</Typography.Text>
                    <Typography.Text
                      css={{
                        fontFamily: 'monospace',
                        backgroundColor: theme.colors.tagDefault,
                        padding: `2px ${theme.spacing.xs}px`,
                        borderRadius: theme.general.borderRadiusBase,
                      }}
                    >
                      {value}
                    </Typography.Text>
                  </div>
                ),
              )}
            </div>
          </div>
        )}

        {/* Created info */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Created:" description="Secret created label" />
          </Typography.Text>
          <TimeAgo date={timestampToDate(secret.created_at)} />
          {secret.created_by && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="by {user}"
                description="Created by user"
                values={{ user: secret.created_by }}
              />
            </Typography.Text>
          )}
        </div>

        {/* Last updated info */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Last Updated:" description="Secret last updated label" />
          </Typography.Text>
          <TimeAgo date={timestampToDate(secret.last_updated_at)} />
          {secret.last_updated_by && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="by {user}"
                description="Updated by user"
                values={{ user: secret.last_updated_by }}
              />
            </Typography.Text>
          )}
        </div>

        {/* Secret ID */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Secret ID:" description="Secret ID label" />
          </Typography.Text>
          <Typography.Text
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {secret.secret_id}
          </Typography.Text>
        </div>
      </div>
    </div>
  );

  if (!showCard) {
    return content;
  }

  return (
    <div
      css={{
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      {content}
    </div>
  );
};
