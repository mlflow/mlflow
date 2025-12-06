import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { formatProviderName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import type { Secret } from '../../types';

interface SecretDetailsProps {
  secret: Secret;
  /** Whether to show as a card with background (default: true) */
  showCard?: boolean;
}

/**
 * Displays detailed information about a secret/API key.
 * Can be used standalone or within a drawer/modal.
 */
export const SecretDetails = ({ secret, showCard = true }: SecretDetailsProps) => {
  const { theme } = useDesignSystemTheme();

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

        {/* Masked value */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary">
            <FormattedMessage defaultMessage="Masked Key:" description="Masked API key label" />
          </Typography.Text>
          <Typography.Text
            css={{
              fontFamily: 'monospace',
              backgroundColor: theme.colors.tagDefault,
              padding: `2px ${theme.spacing.xs}px`,
              borderRadius: theme.general.borderRadiusBase,
            }}
          >
            {secret.masked_value}
          </Typography.Text>
        </div>

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
