import { useMemo } from 'react';
import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { formatProviderName, formatAuthMethodName, formatCredentialFieldName } from '../../utils/providerUtils';
import { timestampToDate } from '../../utils/dateUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { MaskedValueDisplay } from './MaskedValueDisplay';
import type { SecretInfo } from '../../types';

interface SecretDetailsProps {
  secret: SecretInfo;
  showCard?: boolean;
}

export const SecretDetails = ({ secret, showCard = true }: SecretDetailsProps) => {
  const { theme } = useDesignSystemTheme();

  const authConfig = useMemo(() => {
    return secret.auth_config ?? null;
  }, [secret.auth_config]);

  const rowStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
  };

  const labelStyle = {
    minWidth: 100,
    flexShrink: 0,
  };

  const content = (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <Typography.Title level={3} css={{ margin: 0 }}>
        {secret.secret_name}
      </Typography.Title>
      <Spacer size="md" />

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {secret.provider && (
          <div css={rowStyle}>
            <Typography.Text color="secondary" css={labelStyle}>
              <FormattedMessage defaultMessage="Provider" description="Secret provider label" />
            </Typography.Text>
            <Typography.Text>{formatProviderName(secret.provider)}</Typography.Text>
          </div>
        )}

        {authConfig?.['auth_mode'] && (
          <div css={rowStyle}>
            <Typography.Text color="secondary" css={labelStyle}>
              <FormattedMessage defaultMessage="Auth Type" description="Auth type label" />
            </Typography.Text>
            <Typography.Text>{formatAuthMethodName(String(authConfig['auth_mode']))}</Typography.Text>
          </div>
        )}

        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
          <Typography.Text color="secondary" css={labelStyle}>
            <FormattedMessage defaultMessage="Masked Key" description="Masked API key label" />
          </Typography.Text>
          <MaskedValueDisplay maskedValue={secret.masked_values} />
        </div>

        {authConfig && Object.keys(authConfig).filter((k) => k !== 'auth_mode').length > 0 && (
          <div css={rowStyle}>
            <Typography.Text color="secondary" css={labelStyle}>
              <FormattedMessage defaultMessage="Config" description="Auth config label" />
            </Typography.Text>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              {Object.entries(authConfig)
                .filter(([key]) => key !== 'auth_mode')
                .map(([key, value]) => (
                  <div key={key} css={{ display: 'flex', gap: theme.spacing.xs }}>
                    <Typography.Text color="secondary">{formatCredentialFieldName(key)}:</Typography.Text>
                    <Typography.Text css={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {String(value)}
                    </Typography.Text>
                  </div>
                ))}
            </div>
          </div>
        )}

        <div css={rowStyle}>
          <Typography.Text color="secondary" css={labelStyle}>
            <FormattedMessage defaultMessage="Created" description="Secret created label" />
          </Typography.Text>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
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
        </div>

        <div css={rowStyle}>
          <Typography.Text color="secondary" css={labelStyle}>
            <FormattedMessage defaultMessage="Updated" description="Secret last updated label" />
          </Typography.Text>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
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
