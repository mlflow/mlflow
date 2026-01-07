import { useMemo } from 'react';
import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { formatProviderName, formatAuthMethodName, formatCredentialFieldName } from '../../utils/providerUtils';
import { TimeAgo } from '../../../shared/web-shared/browse/TimeAgo';
import { MaskedValueDisplay } from './MaskedValueDisplay';
import type { SecretInfo } from '../../types';

type Theme = ReturnType<typeof useDesignSystemTheme>['theme'];

const getStyles = (theme: Theme) => ({
  labelStyle: {
    minWidth: 100,
    flexShrink: 0,
  } as const,
  rowStyle: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
  } as const,
  rowStyleFlexStart: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: theme.spacing.sm,
  } as const,
  containerStyle: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing.sm,
  } as const,
  configColumnStyle: {
    display: 'flex',
    flexDirection: 'column',
    gap: theme.spacing.xs,
  } as const,
  configRowStyle: {
    display: 'flex',
    gap: theme.spacing.xs,
  } as const,
  timestampRowStyle: {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.xs,
  } as const,
  cardStyle: {
    padding: theme.spacing.md,
    border: `1px solid ${theme.colors.borderDecorative}`,
    borderRadius: theme.general.borderRadiusBase,
    backgroundColor: theme.colors.backgroundSecondary,
  } as const,
  monoStyle: {
    fontFamily: 'monospace',
    wordBreak: 'break-all',
  } as const,
});

interface SecretDetailsProps {
  secret: SecretInfo;
  showCard?: boolean;
}

export const SecretDetails = ({ secret, showCard = true }: SecretDetailsProps) => {
  const { theme } = useDesignSystemTheme();
  const styles = useMemo(() => getStyles(theme), [theme]);

  const authConfig = useMemo(() => {
    return secret.auth_config ?? null;
  }, [secret.auth_config]);

  const content = (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <Typography.Title level={3} css={{ margin: 0 }}>
        {secret.secret_name}
      </Typography.Title>
      <Spacer size="md" />

      <div css={styles.containerStyle}>
        {secret.provider && (
          <div css={styles.rowStyle}>
            <Typography.Text color="secondary" css={styles.labelStyle}>
              <FormattedMessage defaultMessage="Provider" description="Secret provider label" />
            </Typography.Text>
            <Typography.Text>{formatProviderName(secret.provider)}</Typography.Text>
          </div>
        )}

        {authConfig?.['auth_mode'] && (
          <div css={styles.rowStyle}>
            <Typography.Text color="secondary" css={styles.labelStyle}>
              <FormattedMessage defaultMessage="Auth Type" description="Auth type label" />
            </Typography.Text>
            <Typography.Text>{formatAuthMethodName(String(authConfig['auth_mode']))}</Typography.Text>
          </div>
        )}

        <div css={styles.rowStyleFlexStart}>
          <Typography.Text color="secondary" css={styles.labelStyle}>
            <FormattedMessage defaultMessage="Masked Key" description="Masked API key label" />
          </Typography.Text>
          <MaskedValueDisplay maskedValue={secret.masked_values} />
        </div>

        {authConfig && Object.keys(authConfig).filter((k) => k !== 'auth_mode').length > 0 && (
          <div css={styles.rowStyle}>
            <Typography.Text color="secondary" css={styles.labelStyle}>
              <FormattedMessage defaultMessage="Config" description="Auth config label" />
            </Typography.Text>
            <div css={styles.configColumnStyle}>
              {Object.entries(authConfig)
                .filter(([key]) => key !== 'auth_mode')
                .map(([key, value]) => (
                  <div key={key} css={styles.configRowStyle}>
                    <Typography.Text color="secondary">{formatCredentialFieldName(key)}:</Typography.Text>
                    <Typography.Text css={styles.monoStyle}>{String(value)}</Typography.Text>
                  </div>
                ))}
            </div>
          </div>
        )}

        <div css={styles.rowStyle}>
          <Typography.Text color="secondary" css={styles.labelStyle}>
            <FormattedMessage defaultMessage="Created" description="Secret created label" />
          </Typography.Text>
          <div css={styles.timestampRowStyle}>
            <TimeAgo date={new Date(secret.created_at)} />
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

        <div css={styles.rowStyle}>
          <Typography.Text color="secondary" css={styles.labelStyle}>
            <FormattedMessage defaultMessage="Updated" description="Secret last updated label" />
          </Typography.Text>
          <div css={styles.timestampRowStyle}>
            <TimeAgo date={new Date(secret.last_updated_at)} />
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

  return <div css={styles.cardStyle}>{content}</div>;
};
